import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix, vstack as sparse_vstack

# reference, with some modifications https://github.com/shiftbioscience/diversity_by_design/blob/main/analyses/common.py#L148
def calculate_pert_normalized_abs_score_vsrest(
    scores_df_vsrest: pd.DataFrame, 
    names_df_vsrest: pd.DataFrame, 
    var_names: pd.Index,
    control_label: str = 'non-targeting', 
    pert_key: str = 'target_gene'
):
    pert_normalized_abs_scores_vsrest = {}
    for pert in tqdm(scores_df_vsrest.columns, desc="Calculating WMSE Weights"):
        if pert == control_label: # Typically no scores for control in vsrest, but good to check
            continue

        abs_scores = np.abs(scores_df_vsrest[pert].values) # Ensure it's a numpy array
        min_val = np.min(abs_scores)
        max_val = np.max(abs_scores)
        
        if max_val == min_val:
            if max_val == 0: # All scores are 0
                normalized_weights = np.zeros_like(abs_scores)
            else: # All scores are the same non-zero value
                # Squaring ones will still be ones, which is fine.
                normalized_weights = np.ones_like(abs_scores) 
        else:
            normalized_weights = (abs_scores - min_val) / (max_val - min_val)
        
        # Ensure no NaNs in weights, replace with 0 if any (e.g. if a gene had NaN score originally)
        normalized_weights = np.nan_to_num(normalized_weights, nan=0.0)
        
        # Make weighting stronger by squaring the normalized weights
        normalized_weights = np.square(normalized_weights)
        
        weights = pd.Series(normalized_weights, index=names_df_vsrest[pert].values, name=pert)
        # Order by the var_names
        weights = weights.reindex(var_names)
        pert_normalized_abs_scores_vsrest[pert] = weights
    return pert_normalized_abs_scores_vsrest

def get_pert_means(adata, pert_key='target_gene'):
    perturbations = adata.obs[pert_key].unique()
    pert_means = {}
    for pert in tqdm(perturbations, desc="Calculating perturbation means"):
        pert_cells_idx = np.where(adata.obs[pert_key] == pert)[0]
        pert_counts = adata.X[pert_cells_idx].toarray()
        pert_means[pert] = np.mean(pert_counts, axis=0)
    return pert_means

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


# https://github.com/shiftbioscience/diversity_by_design/blob/main/data/norman19/get_data.py
def compute_degs(adata, 
                 control_label = "non-targeting",
                 pert_label = "target_gene",
                 mode='vsrest', 
                 pval_threshold=0.05):
    """
    Compute differentially expressed genes (DEGs) for each perturbation.
    
    Args:
        adata: AnnData object with processed data
        mode: 'vsrest' or 'vscontrol'
            - 'vsrest': Compare each perturbation vs all other perturbations (excluding control)
            - 'vscontrol': Compare each perturbation vs control only
        pval_threshold: P-value threshold for significance (default: 0.05)
    
    Returns:
        dict: rank_genes_groups results dictionary
        
    Adds to adata.uns:
        - deg_dict_{mode}: Dictionary with perturbation as key and dict with 'up'/'down' DEGs as values
        - rank_genes_groups_{mode}: Full rank_genes_groups results
    """
    if mode == 'vsrest':
        # Remove control cells for vsrest analysis
        adata_subset = adata[adata.obs[pert_label] != control_label].copy()
        reference = 'rest'
    elif mode == 'vscontrol':
        # Use full dataset for vscontrol analysis
        adata_subset = adata.copy()
        reference = control_label
    else:
        raise ValueError("mode must be 'vsrest' or 'vscontrol'")
    
    # Compute DEGs
    sc.tl.rank_genes_groups(adata_subset, pert_label, method='t-test_overestim_var', reference=reference)
    
    # Extract results
    names_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["names"])
    pvals_adj_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["pvals_adj"])
    logfc_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["logfoldchanges"])
    
    # For each perturbation, get the significant DEGs up and down regulated
    deg_dict = {}
    for pert in tqdm(adata_subset.obs[pert_label].unique(), desc=f"Computing DEGs {mode}"):
        if mode == 'vscontrol' and pert == control_label:
            continue  # Skip control when comparing vs control
            
        pert_degs = names_df[pert]
        pert_pvals = pvals_adj_df[pert]
        pert_logfc = logfc_df[pert]
        
        # Get significant DEGs
        significant_mask = pert_pvals < pval_threshold
        pert_degs_sig = pert_degs[significant_mask]
        pert_logfc_sig = pert_logfc[significant_mask]
        
        # Split into up and down regulated
        pert_degs_sig_up = pert_degs_sig[pert_logfc_sig > 0].tolist()
        pert_degs_sig_down = pert_degs_sig[pert_logfc_sig < 0].tolist()
        
        deg_dict[pert] = {'up': pert_degs_sig_up, 'down': pert_degs_sig_down}
    
    # Save results to adata.uns
    adata.uns[f'deg_dict_{mode}'] = deg_dict
    adata.uns[f'rank_genes_groups_{mode}'] = adata_subset.uns['rank_genes_groups'].copy()
    
    return adata_subset.uns['rank_genes_groups']
