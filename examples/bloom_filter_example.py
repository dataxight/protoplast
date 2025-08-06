"""
Example usage of the bloom filter functionality for h5ad files.

This example demonstrates how to:
1. Build a bloom filter from an h5ad file
2. Check membership of (cell_index, gene_index) pairs
3. Compare bloom filter results with exact results
"""

import tempfile
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

from protoplast.genomics.anndata import (
    build_bloom_filter_from_h5ad,
    check_pairs_in_h5ad,
    CellGeneBloomFilter
)


def create_example_h5ad():
    """Create an example h5ad file with sparse single-cell data."""
    print("Creating example h5ad file...")
    
    # Create a larger sparse matrix for demonstration
    n_obs = 1000  # 1000 cells
    n_vars = 2000  # 2000 genes
    
    # Create random sparse data (approximately 5% sparsity)
    np.random.seed(42)
    nnz = int(n_obs * n_vars * 0.05)  # 5% non-zero entries
    
    # Generate random indices for non-zero entries
    cell_indices = np.random.randint(0, n_obs, nnz)
    gene_indices = np.random.randint(0, n_vars, nnz)
    values = np.random.exponential(scale=1.0, size=nnz).astype(np.float32)
    
    # Remove duplicates and create CSR matrix
    unique_pairs = {}
    for i, (cell_idx, gene_idx, val) in enumerate(zip(cell_indices, gene_indices, values)):
        if (cell_idx, gene_idx) not in unique_pairs:
            unique_pairs[(cell_idx, gene_idx)] = val
    
    # Convert to lists for CSR matrix construction
    rows, cols, data = zip(*[(cell, gene, val) for (cell, gene), val in unique_pairs.items()])
    
    # Create sparse matrix
    X = csr_matrix((data, (rows, cols)), shape=(n_obs, n_vars))
    
    # Create AnnData object
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    # Save to temporary file
    tmp_file = tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False)
    adata.write_h5ad(tmp_file.name)
    
    print(f"Created h5ad file: {tmp_file.name}")
    print(f"Matrix shape: {X.shape}")
    print(f"Number of non-zero entries: {X.nnz}")
    print(f"Sparsity: {X.nnz / (n_obs * n_vars):.3%}")
    
    return tmp_file.name, list(unique_pairs.keys())


def demonstrate_bloom_filter(h5ad_path, known_pairs):
    """Demonstrate bloom filter functionality."""
    print("\n" + "="*60)
    print("BLOOM FILTER DEMONSTRATION")
    print("="*60)
    
    # Build bloom filter
    print("\n1. Building bloom filter from h5ad file...")
    bloom_filter = build_bloom_filter_from_h5ad(h5ad_path, false_positive_rate=0.01)
    
    print(f"   Filter size: {bloom_filter.size}")
    print(f"   Number of hash functions: {bloom_filter.num_hashes}")
    print(f"   Theoretical false positive rate: {bloom_filter.false_positive_rate(len(known_pairs)):.4f}")
    
    # Test with known existing pairs
    print("\n2. Testing with known existing pairs...")
    test_pairs = known_pairs[:10]  # Test first 10 pairs
    bloom_results = [bloom_filter.contains(cell, gene) for cell, gene in test_pairs]
    
    print(f"   Tested {len(test_pairs)} known existing pairs")
    print(f"   Bloom filter found: {sum(bloom_results)}/{len(bloom_results)} pairs")
    print(f"   (Should be {len(bloom_results)}/{len(bloom_results)} - no false negatives)")
    
    # Test with random non-existing pairs
    print("\n3. Testing with random non-existing pairs...")
    random_pairs = [(np.random.randint(0, 1000), np.random.randint(0, 2000)) for _ in range(100)]
    # Filter out any that might actually exist
    test_nonexistent = [pair for pair in random_pairs if pair not in known_pairs][:20]
    
    bloom_results_nonexistent = [bloom_filter.contains(cell, gene) for cell, gene in test_nonexistent]
    false_positives = sum(bloom_results_nonexistent)
    
    print(f"   Tested {len(test_nonexistent)} random pairs")
    print(f"   Bloom filter found: {false_positives}/{len(test_nonexistent)} pairs")
    print(f"   False positive rate: {false_positives/len(test_nonexistent):.4f}")
    
    # Compare with exact method
    print("\n4. Comparing bloom filter vs exact method...")
    exact_results = check_pairs_in_h5ad(h5ad_path, test_pairs, use_bloom_filter=False)
    bloom_batch_results = check_pairs_in_h5ad(h5ad_path, test_pairs, use_bloom_filter=True)
    
    print(f"   Exact method found: {sum(exact_results)}/{len(exact_results)} pairs")
    print(f"   Bloom filter found: {sum(bloom_batch_results)}/{len(bloom_batch_results)} pairs")
    print(f"   Agreement: {sum(e == b for e, b in zip(exact_results, bloom_batch_results))}/{len(exact_results)}")


def demonstrate_performance_characteristics():
    """Demonstrate different bloom filter configurations."""
    print("\n" + "="*60)
    print("PERFORMANCE CHARACTERISTICS")
    print("="*60)
    
    # Test different filter sizes and false positive rates
    test_cases = [
        (1000, 0.1),   # Small filter, high FP rate
        (10000, 0.01), # Medium filter, low FP rate
        (100000, 0.001), # Large filter, very low FP rate
    ]
    
    for num_items in [100, 1000, 10000]:
        print(f"\nFor {num_items} items:")
        for fp_rate in [0.1, 0.01, 0.001]:
            optimal_size = CellGeneBloomFilter.optimal_size(num_items, fp_rate)
            optimal_hashes = CellGeneBloomFilter.optimal_num_hashes(optimal_size, num_items)
            
            print(f"  FP rate {fp_rate:>6.3f}: size={optimal_size:>8}, hashes={optimal_hashes}")


def main():
    """Main demonstration function."""
    print("Single-Cell Bloom Filter Demonstration")
    print("="*50)
    
    # Create example data
    h5ad_path, known_pairs = create_example_h5ad()
    
    try:
        # Demonstrate bloom filter functionality
        demonstrate_bloom_filter(h5ad_path, known_pairs)
        
        # Show performance characteristics
        demonstrate_performance_characteristics()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("✅ Successfully demonstrated bloom filter functionality")
        print("✅ Showed comparison between bloom filter and exact methods")
        print("✅ Illustrated performance characteristics")
        print("\nKey benefits of bloom filters for single-cell data:")
        print("• Memory efficient for large sparse matrices")
        print("• Fast membership queries O(k) where k is number of hash functions")
        print("• No false negatives (if it says 'no', it's definitely not there)")
        print("• Tunable false positive rate")
        print("• Particularly useful for pre-filtering before expensive operations")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import os
        try:
            os.unlink(h5ad_path)
            print(f"\nCleaned up temporary file: {h5ad_path}")
        except:
            pass


if __name__ == "__main__":
    main()