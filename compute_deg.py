import anndata as ad
import scanpy as sc
import glob
import os
import numpy as np
import pandas as pd

from typing import List
from adata_utils import compute_degs
import gc

def main(file_paths: List[str], 
    output_dir: str,
    control_label: str = 'non-targeting',
    pert_label: str = 'target_gene',
    n_top_genes: int = 8192, 
    max_cells: int = 256, 
    max_cells_control: int = 8192,
    ):
    for i, file_path in enumerate(file_paths):
        # ignore some success files before
        if i < 8:
            continue
        print(f"Processing {file_path} {i}/{len(file_paths)}")
        basename = os.path.basename(file_path)
        adata = ad.read_h5ad(file_path)
        # Downsample each perturbation to have no more than N cells
        print("Downsampling each perturbation to have no more than N cells")

        pert_counts = adata.obs[pert_label].value_counts()
        pert_counts = pert_counts[pert_counts > max_cells]
        cells_to_keep = []

        for pert in pert_counts.index:
            pert_cells = adata.obs[adata.obs[pert_label] == pert].index.tolist()
            if pert == control_label:
                pert_cells = np.random.choice(pert_cells, size=min(len(pert_cells), max_cells_control), replace=False)
            else:
                pert_cells = np.random.choice(pert_cells, size=min(len(pert_cells), max_cells), replace=False)
            cells_to_keep.extend(pert_cells)

        # Subset the adata object
        adata = adata[cells_to_keep]
        print("Computing DEGs")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
        compute_degs(adata, mode='vsrest')
        print("Saving DEGs")
        SCORE_TYPE = 'scores' # or 'logfoldchanges'
        names_df_vsrest = pd.DataFrame(adata.uns["rank_genes_groups_vsrest"]["names"])
        scores_df_vsrest = pd.DataFrame(adata.uns["rank_genes_groups_vsrest"][SCORE_TYPE])
        # Save dataframes to csv
        names_df_vsrest.to_pickle(f'{output_dir}/{basename}_names_df_vsrest.pkl')
        scores_df_vsrest.to_pickle(f'{output_dir}/{basename}_scores_df_vsrest.pkl')
        gc.collect()

if __name__ == "__main__":
    files = glob.glob("/mnt/hdd2/tan/competition_support_set_sorted/*.h5")
    main(files, output_dir="/mnt/hdd2/tan/competition_support_set_sorted/degs")
