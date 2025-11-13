import os
from pathlib import Path  # Import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


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
def test_inference(tmp_path: Path):
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
    checkpoint_path = (tmp_path / "results").rglob("checkpoint.ckpt")
    assert checkpoint_path is not None
    checkpoint_path = list(checkpoint_path)[0]
    infer_test_path = tmp_path / "inference_test.h5ad"
    # use a different seed for different data
    _simulate_h5ad(n_cells=4000, n_genes=200, output_path=infer_test_path, seed=2509)
    result_path = str(tmp_path / "inference_results")
    os.makedirs(result_path, exist_ok=True)
    trainer.inference(
        file_paths=[str(infer_test_path)],
        ckpt_path=str(checkpoint_path),
        result_storage_path=result_path,
        prediction_format="csv"
    )
    assert len([f for f in os.listdir(result_path) if f.endswith(".csv")]) > 0
    result_path = str(tmp_path / "inference_par_results")
    os.makedirs(result_path, exist_ok=True)
    trainer.par_inference(
        file_paths=[str(infer_test_path)],
        ckpt_path=str(checkpoint_path),
        result_storage_path=result_path,
    )
    assert len(os.listdir(result_path)) > 0
    result_path = str(tmp_path / "inference_results_pq")
    os.makedirs(result_path, exist_ok=True)
    trainer.inference(
        file_paths=[str(infer_test_path)],
        ckpt_path=str(checkpoint_path),
        result_storage_path=result_path,
        prediction_format="parquet"
    )
    assert len([f for f in os.listdir(result_path) if f.endswith(".parquet")]) > 0
