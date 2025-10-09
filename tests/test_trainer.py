import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import os

def _simulate_h5ad(n_cells: int, n_genes: int, seed: int=2409) -> str:
    """Simulate a AnnData with 2 cell lines and return a path to the h5ad file"""

    # Set seed for reproducibility
    np.random.seed(seed)

    # Create sparse expression matrix (CSR)
    # Random data with sparsity
    X = sp.random(n_cells, n_genes, density=0.1, format="csr", data_rvs=np.random.rand)

    # Create obs with cell_line column (50% each)
    cell_lines = np.array(["CellLineA"] * (n_cells // 2) + ["CellLineB"] * (n_cells - n_cells // 2))
    obs = pd.DataFrame({"cell_line": cell_lines}, index=[f"cell_{i}" for i in range(n_cells)])

    # Create var with gene names
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Write data
    os.makedirs("tmp", exist_ok=True)
    adata.write_h5ad("tmp/test_trainer.h5ad")
    return "tmp/test_trainer.h5ad"

def test_trainer():
    """Test if RayTrainRunner can be initialized with default parameters"""
    test_h5ad = _simulate_h5ad(n_cells=5000, n_genes=200)

    from protoplast import DistributedCellLineAnnDataset, LinearClassifier, RayTrainRunner

    trainer = RayTrainRunner(
        Ds=DistributedCellLineAnnDataset,
        Model=LinearClassifier,
        model_keys=["num_genes", "num_classes"],
    )
    assert isinstance(trainer, RayTrainRunner)
    trainer.train([test_h5ad])

