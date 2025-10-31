from pathlib import Path  # Import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch


# Helper function is modified to accept an output path
def _simulate_h5ad(n_cells: int, n_genes: int, output_path: Path, seed: int = 2409):
    """Simulate an AnnData object and write it to the provided path."""
    np.random.seed(seed)
    X = sp.random(n_cells, n_genes, density=0.1, format="csr", data_rvs=np.random.rand)
    cell_lines = np.array(["CellLineA"] * (n_cells // 2) + ["CellLineB"] * (n_cells - n_cells // 2))
    obs = pd.DataFrame({"cell_line": cell_lines}, index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(output_path)


# The test function now accepts the tmp_path fixture
def test_trainer(tmp_path: Path):
    """Test if RayTrainRunner can be initialized with default parameters."""
    test_h5ad_path = tmp_path / "test_trainer.h5ad"
    _simulate_h5ad(n_cells=5000, n_genes=200, output_path=test_h5ad_path)

    from protoplast import DistributedCellLineAnnDataset, LinearClassifier, RayTrainRunner

    trainer = RayTrainRunner(
        Ds=DistributedCellLineAnnDataset,
        Model=LinearClassifier,
        model_keys=["num_genes", "num_classes"],
    )
    assert isinstance(trainer, RayTrainRunner)
    trainer.train([str(test_h5ad_path)], result_storage_path=str(tmp_path / "results"))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() <= 1,
    reason="This test requires CUDA and more than 1 GPU.",
)
def test_trainer_multi_gpu(tmp_path: Path):
    """Test if RayTrainRunner can be initialized with multi-GPU setup."""
    device_count = torch.cuda.device_count()
    print(f"✅ Running multi-GPU test. Total: {device_count} GPUs.")
    test_h5ad_path = tmp_path / "test_trainer_multi_gpu.h5ad"
    _simulate_h5ad(n_cells=5000, n_genes=200, output_path=test_h5ad_path)

    from protoplast import DistributedCellLineAnnDataset, LinearClassifier, RayTrainRunner

    trainer = RayTrainRunner(
        Ds=DistributedCellLineAnnDataset,
        Model=LinearClassifier,
        model_keys=["num_genes", "num_classes"],
    )
    assert isinstance(trainer, RayTrainRunner)
    trainer.train(
        [str(test_h5ad_path)], result_storage_path=str(tmp_path / "results_multi_gpu"), num_workers=device_count
    )
