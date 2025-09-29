import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix, vstack as sparse_vstack

def fix_genes_chunked(adata, fixed_genes, chunk_size=10000):
    """
    Chunked implementation: iterate rows (cells) in blocks of `chunk_size`.
    For each chunk select present columns, map them to final column indices, and append.
    """
    fixed_genes = pd.Index(fixed_genes)
    var_new = adata.var.reindex(fixed_genes).copy()

    present_mask = fixed_genes.isin(adata.var_names)
    present_idx = adata.var_names.get_indexer(fixed_genes[present_mask])
    present_positions = np.where(present_mask)[0]  # final column indices for present genes

    n_cells = adata.n_obs
    n_genes = len(fixed_genes)

    is_sparse = sparse.issparse(adata.X)

    chunks = []  # store chunk matrices (dense np arrays or sparse csr matrices)
    dtype = adata.X.dtype

    for chunk_i, start in enumerate(range(0, n_cells, chunk_size)):
        print(f"Processing chunk {chunk_i}/{int(n_cells / chunk_size)}")
        stop = min(start + chunk_size, n_cells)
        rows = slice(start, stop)
        n_chunk = stop - start

        if is_sparse:
            if present_positions.size == 0:
                # Entire chunk zeros
                chunk_full = csr_matrix((n_chunk, n_genes), dtype=dtype)
            else:
                # select rows and present columns from original
                chunk_present = adata.X[rows, :][:, present_idx]  # sparse (n_chunk, n_present)
                if chunk_present.nnz == 0:
                    chunk_full = csr_matrix((n_chunk, n_genes), dtype=dtype)
                else:
                    coo = chunk_present.tocoo()
                    # map local chunk column index -> final column index
                    cols_mapped = present_positions[coo.col]
                    chunk_full = coo_matrix((coo.data, (coo.row, cols_mapped)),
                                            shape=(n_chunk, n_genes), dtype=dtype).tocsr()
        else:
            # dense flow
            chunk = np.asarray(adata.X[rows, :])  # (n_chunk, original_genes)
            chunk_full = np.zeros((n_chunk, n_genes), dtype=dtype)
            if present_positions.size > 0:
                chunk_present = chunk[:, present_idx]  # (n_chunk, n_present)
                chunk_full[:, present_positions] = chunk_present

        chunks.append(chunk_full)

    # stack chunks back together
    if is_sparse:
        X_new = sparse_vstack(chunks).tocsr()
    else:
        X_new = np.vstack(chunks)

    return ad.AnnData(X_new, obs=adata.obs.copy(), var=var_new)
