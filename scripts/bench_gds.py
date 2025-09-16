import torch
import numpy as np
from scipy import sparse
import os
import h5py
import time


def create_sparse_csr_matrix(n_rows, n_cols, density=0.01, seed=42):
    """Create a random sparse CSR matrix using scipy."""
    np.random.seed(seed)
    # Create random sparse matrix
    matrix = sparse.random(n_rows, n_cols, density=density, format='csr', dtype=np.float32)
    return matrix


def save_csr_to_h5(csr_matrix, filename):
    """
    Save CSR matrix components to HDF5 file.
    
    Returns:
        dict: Metadata containing matrix information
    """
    with h5py.File(filename, 'w') as f:
        # Save CSR components without compression
        f.create_dataset('indptr', data=csr_matrix.indptr)
        f.create_dataset('indices', data=csr_matrix.indices)
        f.create_dataset('data', data=csr_matrix.data)
        
        # Save metadata
        f.attrs['n_rows'] = csr_matrix.shape[0]
        f.attrs['n_cols'] = csr_matrix.shape[1]
        f.attrs['nnz'] = csr_matrix.nnz
    
    metadata = {
        'n_rows': csr_matrix.shape[0],
        'n_cols': csr_matrix.shape[1],
        'nnz': csr_matrix.nnz
    }
    
    return metadata


def load_csr_range_from_h5(filename, start_row, end_row, device='cuda'):
    """
    Load a range of rows [start_row:end_row, :] from CSR matrix stored in HDF5 file.
    
    Args:
        filename: HDF5 file path
        start_row: Starting row index (inclusive)
        end_row: Ending row index (exclusive)
        device: Target device for the tensor ('cuda' or 'cpu')
    
    Returns:
        torch.sparse_csr_tensor: Sparse CSR tensor on specified device
    """
    with h5py.File(filename, 'r') as f:
        n_rows = f.attrs['n_rows']
        n_cols = f.attrs['n_cols']
        
        if start_row < 0 or end_row > n_rows or start_row >= end_row:
            raise ValueError(f"Invalid row range [{start_row}:{end_row}] for matrix with {n_rows} rows")
        
        # Load full indptr to determine data range
        indptr = f['indptr'][:]
        
        # Get data range for selected rows
        data_start_idx = indptr[start_row]
        data_end_idx = indptr[end_row]
        n_data_elements = data_end_idx - data_start_idx
        
        if n_data_elements == 0:
            # Empty slice
            new_indptr = np.zeros(end_row - start_row + 1, dtype=np.int32)
            empty_indices = np.empty(0, dtype=np.int32)
            empty_data = np.empty(0, dtype=np.float32)
            
            # Create sparse CSR tensor on CPU first
            sparse_tensor = torch.sparse_csr_tensor(
                torch.from_numpy(new_indptr),
                torch.from_numpy(empty_indices),
                torch.from_numpy(empty_data),
                size=(end_row - start_row, n_cols)
            )
            
            # Move to target device
            sparse_tensor = sparse_tensor.to(device)
            
            return sparse_tensor
        
        # Load relevant data slices
        slice_indices = f['indices'][data_start_idx:data_end_idx]
        slice_data = f['data'][data_start_idx:data_end_idx]
        slice_indptr = indptr[start_row:end_row + 1] - data_start_idx
        
        # Convert to numpy arrays with correct dtypes
        slice_indptr = slice_indptr.astype(np.int32)
        slice_indices = slice_indices.astype(np.int32)
        slice_data = slice_data.astype(np.float32)
        
        # Create sparse CSR tensor on CPU first
        sparse_tensor = torch.sparse_csr_tensor(
            torch.from_numpy(slice_indptr),
            torch.from_numpy(slice_indices),
            torch.from_numpy(slice_data),
            size=(end_row - start_row, n_cols)
        )
        
        # Move to target device
        sparse_tensor = sparse_tensor.to(device)
    
    return sparse_tensor
def save_csr_to_gds(csr_matrix, filename):
    """
    Save CSR matrix components (indptr, indices, data) to a CUDA GDS file.
    
    Returns:
        dict: Metadata containing offsets (in bytes) and element counts for each component.
    """
    # Convert scipy CSR components to CUDA tensors
    indptr_tensor = torch.from_numpy(csr_matrix.indptr.astype(np.int32)).cuda()
    indices_tensor = torch.from_numpy(csr_matrix.indices.astype(np.int32)).cuda()
    data_tensor = torch.from_numpy(csr_matrix.data.astype(np.float32)).cuda()
    
    # Create/overwrite GDS file
    file = torch.cuda.gds.GdsFile(filename, os.O_CREAT | os.O_RDWR)
    
    # Calculate byte offsets
    indptr_offset = 0
    indices_offset = indptr_tensor.nbytes
    data_offset = indices_offset + indices_tensor.nbytes
    
    # Save components sequentially
    file.save_storage(indptr_tensor.untyped_storage(), offset=indptr_offset)
    file.save_storage(indices_tensor.untyped_storage(), offset=indices_offset)
    file.save_storage(data_tensor.untyped_storage(), offset=data_offset)
    
    # Store and report metadata
    metadata = {
        'n_rows': csr_matrix.shape[0],
        'n_cols': csr_matrix.shape[1],
        'nnz': csr_matrix.nnz,
        'indptr_offset': indptr_offset,
        'indptr_size': len(csr_matrix.indptr),
        'indices_offset': indices_offset,
        'indices_size': len(csr_matrix.indices),
        'data_offset': data_offset,
        'data_size': len(csr_matrix.data),
    }
    
    return metadata


def load_csr_range_from_gds(filename, metadata, start_row, end_row):
    """
    Load a range of rows [start_row:end_row, :] from CSR matrix stored in a GDS file.
    
    Args:
        filename: GDS file path
        metadata: Metadata dict from save_csr_to_gds
        start_row: Starting row index (inclusive)
        end_row: Ending row index (exclusive)
    
    Returns:
        torch.sparse_csr_tensor: Sparse CSR tensor on CUDA
    """
    if start_row < 0 or end_row > metadata['n_rows'] or start_row >= end_row:
        raise ValueError(f"Invalid row range [{start_row}:{end_row}] for matrix with {metadata['n_rows']} rows")
    
    file = torch.cuda.gds.GdsFile(filename, os.O_RDONLY)
    
    # Load full indptr to determine data range for selected rows
    full_indptr = torch.empty(metadata['indptr_size'], dtype=torch.int32, device='cuda')
    file.load_storage(full_indptr.untyped_storage(), offset=metadata['indptr_offset'])
    
    # Get data range for selected rows
    data_start_idx = full_indptr[start_row].item()
    data_end_idx = full_indptr[end_row].item()
    n_data_elements = data_end_idx - data_start_idx
    
    if n_data_elements == 0:
        # Empty slice - create empty sparse tensor
        new_indptr = torch.zeros(end_row - start_row + 1, dtype=torch.int32, device='cuda')
        empty_indices = torch.empty(0, dtype=torch.int32, device='cuda')
        empty_data = torch.empty(0, dtype=torch.float32, device='cuda')
        return torch.sparse_csr_tensor(
            new_indptr, empty_indices, empty_data,
            size=(end_row - start_row, metadata['n_cols'])
        )
    
    # Load relevant indices and data
    slice_indices = torch.empty(n_data_elements, dtype=torch.int32, device='cuda')
    slice_data = torch.empty(n_data_elements, dtype=torch.float32, device='cuda')
    
    # Calculate byte offsets for the data slice
    indices_byte_offset = metadata['indices_offset'] + data_start_idx * 4  # 4 bytes per int32
    data_byte_offset = metadata['data_offset'] + data_start_idx * 4        # 4 bytes per float32
    
    file.load_storage(slice_indices.untyped_storage(), offset=indices_byte_offset)
    file.load_storage(slice_data.untyped_storage(), offset=data_byte_offset)
    
    # Create new indptr for the slice
    slice_indptr = full_indptr[start_row:end_row + 1] - data_start_idx
    
    # Create sparse CSR tensor
    sparse_tensor = torch.sparse_csr_tensor(
        slice_indptr, slice_indices, slice_data,
        size=(end_row - start_row, metadata['n_cols'])
    )
    
    return sparse_tensor


def sparse_csr_equal(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """Return True if two torch CSR tensors are equal (structure and values)."""
    if a.layout != torch.sparse_csr:
        a = a.to_sparse_csr()
    if b.layout != torch.sparse_csr:
        b = b.to_sparse_csr()
    if a.size() != b.size():
        return False
    if not torch.equal(a.crow_indices(), b.crow_indices()):
        return False
    if not torch.equal(a.col_indices(), b.col_indices()):
        return False
    return torch.allclose(a.values(), b.values(), rtol=rtol, atol=atol)

# Example usage and testing
if __name__ == "__main__":
    # Create a test sparse matrix of shape (N, G)
    N, G = 100000, 18080
    density = 0.03
    print(f"Creating sparse CSR matrix of shape ({N}, {G}) with density {density}...")
    csr_matrix = create_sparse_csr_matrix(N, G, density=density)
    assert len(csr_matrix.indptr) == N + 1, "CSR indptr length must be N+1"
    
    # Save to both GDS and H5 files
    gds_filename = "sparse_matrix.gds"
    h5_filename = "sparse_matrix.h5"
    
    print("\nSaving CSR to GDS and H5...")
    gds_metadata = save_csr_to_gds(csr_matrix, gds_filename)
    h5_metadata = save_csr_to_h5(csr_matrix, h5_filename)
    
    # Test loading different row ranges (start:end, G)
    test_ranges = [
        (0, 1000),           # first 1k rows
        (10000, 20000),      # 10k rows
        (50000, 60000),      # middle 10k rows
        (90000, 100000),     # last 10k rows
        (5000, 5001),        # single row
    ]
    
    print("\n=== Performance Comparison ===")
    
    # Store timing results for comparison
    gds_times = {}
    h5_times = {}
    
    for start_row, end_row in test_ranges:
        print(f"Testing rows [{start_row}:{end_row}] ({end_row - start_row} rows)")
        
        # Test GDS loading
        gds_start = time.perf_counter()
        gds_tensor = load_csr_range_from_gds(gds_filename, gds_metadata, start_row, end_row)
        gds_end = time.perf_counter()
        gds_time = gds_end - gds_start
        gds_times[(start_row, end_row)] = gds_time
        
        # Test H5 loading
        h5_start = time.perf_counter()
        h5_tensor = load_csr_range_from_h5(h5_filename, start_row, end_row)
        h5_end = time.perf_counter()
        h5_time = h5_end - h5_start
        h5_times[(start_row, end_row)] = h5_time
        
        # Equality
        equal = sparse_csr_equal(gds_tensor, h5_tensor)
        speedup = h5_time / gds_time if gds_time > 0 else float('inf')
        print(f"  GDS: {gds_time:.4f}s | H5: {h5_time:.4f}s | speedup: {speedup:.2f}x | equal: {'YES' if equal else 'NO'}")
    
    # Summary statistics
    print("\n=== TIMING SUMMARY ===")
    print(f"{'Range':<15} {'Rows':<8} {'GDS (s)':<10} {'H5 (s)':<10} {'Speedup':<10}")
    print("-" * 55)
    
    total_gds_time = 0
    total_h5_time = 0
    
    for (start_row, end_row) in test_ranges:
        n_rows_range = end_row - start_row
        gds_time = gds_times[(start_row, end_row)]
        h5_time = h5_times[(start_row, end_row)]
        speedup = h5_time / gds_time if gds_time > 0 else float('inf')
        
        total_gds_time += gds_time
        total_h5_time += h5_time
        
        print(f"[{start_row}:{end_row}]"[:14].ljust(15) + 
              f"{n_rows_range:<8} {gds_time:<10.4f} {h5_time:<10.4f} {speedup:<10.2f}x")
    
    print("-" * 55)
    avg_speedup = total_h5_time / total_gds_time if total_gds_time > 0 else float('inf')
    print(f"{'AVERAGE':<15} {'—':<8} {total_gds_time:<10.4f} {total_h5_time:<10.4f} {avg_speedup:<10.2f}x")
    
    # # Cleanup
    # if os.path.exists(gds_filename):
    #     os.remove(gds_filename)
    # if os.path.exists(h5_filename):
    #     os.remove(h5_filename)
    # print("\nCleanup complete.")