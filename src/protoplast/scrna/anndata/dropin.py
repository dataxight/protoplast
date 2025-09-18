#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse

import anndata as ad

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def read_h5ad(adata_path: str | Path) -> "AnnDataGDS":
    """
    Reads a `.h5ad` file and returns an `AnnDataGDS` object. If the index files
    (.obs, .var, .gds, .metadata) do not exist, they are created.

    Args:
        adata_path (str | Path): Path to `.h5ad` file.

    Returns:
        AnnDataGDS: Wrapper object providing indexed access to AnnData using GDS.
    """
    adata_path = Path(adata_path)
    assert adata_path.exists(), f"File does not exist: {adata_path}"
    scdata = AnnDataGDS(adata_path)
    if not scdata.has_index():
        scdata.create_index()
    else:
        try:
            scdata.load_index()
        except RuntimeError:
            # Index is stale; recreate
            scdata.delete_index()
            scdata.create_index()
    scdata.load_index()
    return scdata


def save_csr_to_gds(
    adata: "ad.AnnData",
    filename: str | Path,
    batch_rows: int | None = None,
) -> dict[str, Any]:
    """
    Save matrix in CSR-backed format using two GDS files and a global indptr metadata.

    New storage layout (csr_gds_v1):
    - indices.dat (GDS): concatenated csr.indices across batches (int32)
    - data.dat    (GDS): concatenated csr.data across batches (float32)
    - indptr.npy  (CPU): global CSR indptr array (int64) of length n_rows + 1

    Supports two modes for input:
    - Direct sparse matrix: converted to CSR and written in one pass
    - Batched AnnData: reads `adata.X[start:end]` in CSR by batches and streams

    Args:
        adata: an AnnData object
        filename: Base path; sibling files <base>.indices.dat, <base>.data.dat, <base>.indptr.npy are created
        batch_rows: Rows per batch when streaming from AnnData. If None, uses env PROTOPLAST_GDS_BATCH_ROWS or 65536.

    Returns:
        dict: Metadata describing CSR-GDS storage and shapes.
    """
    filename = Path(filename)
    indices_path = filename.with_suffix(".indices.dat")
    data_path = filename.with_suffix(".data.dat")
    indptr_path = filename.with_suffix(".indptr.npy")

    # Helper to obtain a CSR slice for adata rows [start:end)
    def _slice_to_csr(adata_obj: "ad.AnnData", start: int, end: int) -> sparse.csr_matrix:
        X_slice = adata_obj.X[start:end]
        if hasattr(X_slice, "tocsr"):
            return X_slice.tocsr()
        tname = type(X_slice).__name__
        if tname == "_CSRDataset":
            return X_slice.to_memory().tocsr()
        if tname == "Dataset":
            return sparse.csr_matrix(np.array(X_slice))
        # Fallback: treat as dense/sparse array-like
        return sparse.csr_matrix(X_slice)

    adata_obj: ad.AnnData = adata
    n_rows, n_cols = adata_obj.shape
    batch_rows = (
        batch_rows
        if batch_rows is not None
        else int(os.environ.get("PROTOPLAST_GDS_BATCH_ROWS", 65536))
    )

    # Optional progress bars via tqdm, with safe fallback
    try:
        from tqdm import tqdm as _tqdm
        _has_tqdm = True
    except Exception:
        _has_tqdm = False
        def _tqdm(iterable=None, **kwargs):
            return iterable if iterable is not None else None

    # Create/overwrite GDS files
    indices_file = torch.cuda.gds.GdsFile(str(indices_path), os.O_CREAT | os.O_TRUNC | os.O_RDWR)
    data_file = torch.cuda.gds.GdsFile(str(data_path), os.O_CREAT | os.O_TRUNC | os.O_RDWR)

    indices_bytes_written = 0
    data_bytes_written = 0
    total_nnz = 0
    global_indptr = [0]
    _num_batches = (n_rows + batch_rows - 1) // batch_rows
    pbar = _tqdm(total=_num_batches, desc="Writing indices and data", unit="batch") if _has_tqdm else None
    for start in range(0, n_rows, batch_rows):
        end = min(start + batch_rows, n_rows)
        csr_batch = _slice_to_csr(adata_obj, start, end)
        nnz_batch = int(csr_batch.nnz)
        # extend global indptr excluding first 0, offset by total_nnz
        if nnz_batch > 0:
            global_indptr.extend((csr_batch.indptr[1:] + total_nnz).tolist())
        else:
            # all zeros in this batch
            global_indptr.extend([total_nnz] * (end - start))
            continue
        total_nnz += nnz_batch
        cols_np = csr_batch.indices.astype(np.int32, copy=False)
        cols_tensor = torch.from_numpy(cols_np).cuda()
        indices_file.save_storage(cols_tensor.untyped_storage(), offset=indices_bytes_written)
        indices_bytes_written += cols_tensor.nbytes
        data_np = csr_batch.data.astype(np.float32, copy=False)
        data_tensor = torch.from_numpy(data_np).cuda()
        data_file.save_storage(data_tensor.untyped_storage(), offset=data_bytes_written)
        data_bytes_written += data_tensor.nbytes
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    # Save indptr to CPU .npy (int64)
    indptr_arr = np.asarray(global_indptr, dtype=np.int64)
    np.save(indptr_path, indptr_arr)

    # Touch the base marker file so has_index() still works
    try:
        filename.touch(exist_ok=True)
    except Exception:
        pass

    metadata = {
        "format": "csr_gds_v1",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(total_nnz),
        "indices_file": indices_path.name,
        "data_file": data_path.name,
        "indptr_file": indptr_path.name,
        "dtypes": {"indices": "int32", "data": "float32", "indptr": "int64"},
    }
    return metadata

    


def load_range_from_gds(
    filename: str | Path,
    metadata: dict[str, Any], 
    start_row: int, 
    end_row: int,
    format: str = "csr"
) -> torch.Tensor:
    """
    Load a range of rows [start_row:end_row, :] from matrix stored in GDS-backed files.

    Args:
        filename: Base GDS file path (used to resolve sibling files)
        metadata: Metadata dict from save_csr_to_gds
        start_row: Starting row index (inclusive)
        end_row: Ending row index (exclusive)

    Returns:
        torch.sparse_coo_tensor: Sparse COO tensor on CUDA
    """
    # TODO: should not hardcode cuda as device
    filename = Path(filename)

    if start_row < 0 or end_row > metadata["n_rows"] or start_row >= end_row:
        raise ValueError(f"Invalid row range [{start_row}:{end_row}] for matrix with {metadata['n_rows']} rows")

    # If stored in new CSR-GDS format, load only the required slices
    base_dir = filename.parent
    indices_path = base_dir / metadata["indices_file"]
    data_path = base_dir / metadata["data_file"]
    indptr_path = base_dir / metadata["indptr_file"]

    # Memory-map indptr and compute required ranges
    indptr_mmap = np.load(indptr_path)
    row_ptr_start = int(indptr_mmap[start_row])
    row_ptr_end = int(indptr_mmap[end_row])
    nnz_range = row_ptr_end - row_ptr_start

    if nnz_range == 0:
        num_rows = end_row - start_row
        empty_indices = torch.empty((2, 0), dtype=torch.int64, device="cuda")
        empty_values = torch.empty(0, dtype=torch.float32, device="cuda")
        return torch.sparse_coo_tensor(
            empty_indices, empty_values, size=(num_rows, metadata["n_cols"])
        )

    # Load column indices slice
    indices_file = torch.cuda.gds.GdsFile(str(indices_path), os.O_RDONLY)
    cols_tensor = torch.empty(nnz_range, dtype=torch.int32, device="cuda")
    indices_file.load_storage(
        cols_tensor.untyped_storage(),
        offset=row_ptr_start * np.dtype(np.int32).itemsize,
    )

    # Load data slice
    data_file = torch.cuda.gds.GdsFile(str(data_path), os.O_RDONLY)
    data_tensor = torch.empty(nnz_range, dtype=torch.float32, device="cuda")
    data_file.load_storage(
        data_tensor.untyped_storage(),
        offset=row_ptr_start * np.dtype(np.float32).itemsize,
    )

    # Build row indices for COO from indptr slice
    indptr_slice = deepcopy(indptr_mmap[start_row : end_row + 1])
    row_counts = np.diff(indptr_slice).astype(np.int64, copy=False)
    if row_counts.sum() != nnz_range:
        raise RuntimeError("Inconsistent indptr slice sums with nnz range")

    if format == "coo":
        # Create per-element row indices by repeating row id by its nnz count
        row_ids = torch.arange(0, end_row - start_row, device="cuda", dtype=torch.int64)
        row_repeated = row_ids.repeat_interleave(torch.from_numpy(row_counts).to(device="cuda"))

        indices = torch.stack([row_repeated, cols_tensor.long()], dim=0)
        coo_tensor = torch.sparse_coo_tensor(
            indices, data_tensor, size=(end_row - start_row, metadata["n_cols"]) 
            )
        return coo_tensor.coalesce()
    elif format == "csr":
        return torch.sparse_csr_tensor(
            torch.from_numpy(indptr_slice).long().to(device="cuda"),
            cols_tensor.long(),
            data_tensor,
            size=(end_row - start_row, metadata["n_cols"])
        )
    else:
        raise ValueError(f"Unsupported format: {format}")

class AnnDataGDS:
    """
    Drop-in replacement for AnnData that stores X as a sparse COO matrix in a GDS file.
    Supports efficient range-based access returning sparse tensors on GPU with easy striding.
    """

    def __init__(self, adata_path: str | Path) -> None:
        """
        Wrapper around AnnData for fast, chunked access using GDS-stored sparse data.

        Args:
            adata_path (str | Path): Path to `.h5ad` file.
        """
        self.adata_path = Path(adata_path)

        # GDS directory layout
        self.gds_dir = self.adata_path.with_suffix(".gds")
        self.obs_path = self.gds_dir / "obs.pkl"
        self.var_path = self.gds_dir / "var.pkl"
        self.metadata_path = self.gds_dir / "metadata.json"
        self.mtime_path = self.gds_dir / "mtime"
        # Base file prefix for CSR shards
        self.gds_base = self.gds_dir / "matrix"

        # Variables updated later
        self.obs: pd.DataFrame | None = None
        self.var: pd.DataFrame | None = None
        self.gds_metadata: dict[str, Any] | None = None
        self.n_obs: int = 0
        self.n_vars: int = 0
        self.adata_mtime: float = 0.0
        self._matrix_format: str = "csr"

    def has_index(self) -> bool:
        """
        Check whether index files (.obs, .var, .gds, .metadata) exist.

        Returns:
            bool: True if index files exist, False otherwise.
        """
        return (
            self.gds_dir.exists()
            and self.obs_path.exists()
            and self.var_path.exists()
            and self.metadata_path.exists()
            and (self.gds_dir / "matrix.indices.dat").exists()
            and (self.gds_dir / "matrix.data.dat").exists()
            and (self.gds_dir / "matrix.indptr.npy").exists()
        )

    def load_index(self) -> bool:
        """
        Load precomputed index files into memory.

        Returns:
            bool: True if successful.
        """
        assert self.has_index(), "Index files are missing. Run create_index() first."
        self._check_mtime()  # verify file consistency

        logger.info("Loading index: obs")
        self.obs = pd.read_pickle(self.obs_path)

        logger.info("Loading index: var")
        self.var = pd.read_pickle(self.var_path)

        logger.info("Loading index: gds metadata")
        with open(self.metadata_path) as f:
            self.gds_metadata = json.load(f)

        self.n_vars = self.var.shape[0]
        self.n_obs = self.obs.shape[0]

        return True

    def _save_mtime(self) -> None:
        """Save current .h5ad modification time to a file."""
        self.adata_mtime = self.adata_path.stat().st_mtime
        self.mtime_path.write_text(str(self.adata_mtime))

    def _load_mtime(self) -> float:
        """Load stored modification time."""
        if self.mtime_path.exists():
            return float(self.mtime_path.read_text())
        return 0.0

    def _check_mtime(self) -> None:
        """Check that the current file mtime matches the stored mtime."""
        stored = self._load_mtime()
        current = self.adata_path.stat().st_mtime
        if stored != current:
            raise RuntimeError(
                f"The .h5ad file has changed since the index was created.\n"
                f"Stored mtime: {stored}, current mtime: {current}"
            )

    def create_index(self) -> None:
        """
        Create index files by processing the `.h5ad` file and converting X to GDS format.
        """
        logger.info("Creating GDS index")
        logger.info("Reading adata file")
        adata = ad.read_h5ad(str(self.adata_path), backed="r")
        n_obs, n_vars = adata.shape

        # Prepare directory
        self.gds_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving X to GDS file by batches (streaming rows)")
        gds_metadata = save_csr_to_gds(adata, self.gds_base)

        logger.info("Saving GDS metadata")
        with open(self.metadata_path, "w") as f:
            json.dump(gds_metadata, f, indent=2)

        logger.info("Packing obs")
        adata.obs.to_pickle(self.obs_path)

        logger.info("Packing var")
        adata.var.to_pickle(self.var_path)

        self._save_mtime()
        logger.info("GDS index creation complete")

    def delete_index(self) -> None:
        """
        Delete the `.obs`, `.var`, `.gds`, and `.metadata` files created by AnnDataGDS.
        """
        import shutil
        if self.gds_dir.exists():
            shutil.rmtree(self.gds_dir)
            logger.info(f"Deleted: {self.gds_dir}")
        else:
            logger.warning(f"Directory not found, skipping: {self.gds_dir}")

    def set_matrix_format(self, value: str = "csr") -> None:
        """
        Set the X matrix accessor.
        """
        self._matrix_format = value

    @property
    def X(self) -> "GDSMatrixAccessor":
        """
        Provide access to the sparse matrix data via GDS.
        Returns a GDSMatrixAccessor that supports slicing.
        """
        return GDSMatrixAccessor(self.gds_base, self.gds_metadata, self._matrix_format)

    def __getitem__(self, indices: list[int]) -> "ad.AnnData":
        """
        Get an AnnData subset for specific cell indices.
        """
        raise NotImplementedError("__getitem__ is not implemented for this class")

class GDSMatrixAccessor:
    """
    Accessor class for GDS-stored sparse matrix that supports slicing operations.
    """

    def __init__(self, gds_path: Path, metadata: dict[str, Any], matrix_format: str = "csr"):
        self.gds_path = gds_path
        self.metadata = metadata
        self._matrix_format = matrix_format

    @property
    def shape(self) -> tuple:
        """Return the shape of the matrix."""
        return (self.metadata["n_rows"], self.metadata["n_cols"])

    def __getitem__(self, key: int | slice | tuple) -> torch.Tensor:
        """
        Support matrix slicing operations like adata.X[start:end] or adata.X[start:end, :].

        Args:
            key: int, slice, or tuple of (row_slice, col_slice)

        Returns:
            torch.sparse_coo_tensor | torch.Tensor: Sparse CUDA tensor (CSR or COO)
        """
        if isinstance(key, int):
            # Single row access
            return load_range_from_gds(self.gds_path, self.metadata, key, key + 1, self._matrix_format)
        elif isinstance(key, slice):
            # Row slice access
            start, stop, step = key.indices(self.metadata["n_rows"])
            if step != 1:
                raise NotImplementedError("Step slicing not supported")
            return load_range_from_gds(self.gds_path, self.metadata, start, stop, self._matrix_format)
        elif isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key

            # Handle row indexing
            if isinstance(row_key, int):
                start_row, end_row = row_key, row_key + 1
            elif isinstance(row_key, slice):
                start_row, end_row, step = row_key.indices(self.metadata["n_rows"])
                if step != 1:
                    raise NotImplementedError("Step slicing not supported")
            else:
                raise TypeError(f"Unsupported row index type: {type(row_key)}")

            # Load the row range as COO
            sparse_tensor = load_range_from_gds(self.gds_path, self.metadata, start_row, end_row, self._matrix_format)

            # Handle column slicing if needed
            if col_key != slice(None):
                raise NotImplementedError("Column slicing not supported")
            return sparse_tensor
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")

def csr_row_contiguous_view(crow_indices, col_indices, values, shape, start, stop):
    """
    Contiguous row slice [start:stop] as a zero-copy CSR view.
    """
    n_rows, n_cols = shape
    # Adjust crow to start from zero without copying col/values
    crow_new = crow_indices[start:stop+1] - crow_indices[start]
    col_new = col_indices   # same storage
    vals_new = values       # same storage
    shape_new = (stop - start, n_cols)
    return crow_new, col_new, vals_new, shape_new