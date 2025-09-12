import pathlib
from collections.abc import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from protoplast.scrna.anndata.torch_dataloader import (
    AnnDataModule,
    DistributedAnnDataset,
    ann_split_data,
    cell_line_metadata_cb,
    BlockBasedAnnDataset,
)


@pytest.fixture(scope="function")
def test_even_h5ad_file(tmpdir: pathlib.Path) -> str:
    # Create a small AnnData object with sparse data
    # dense matrix:
    # [[1, 0, 2, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 3, 0, 4, 0],
    #  [5, 0, 0, 0, 0]]
    n_obs = 4
    n_vars = 5

    indptr = np.array([0, 2, 2, 4, 5])
    indices = np.array([0, 2, 1, 3, 0])
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    X = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)], data={"cell_line": pd.Categorical([0, 0, 1, 1])})
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    filepath = tmpdir / "test.h5ad"
    adata.write_h5ad(filepath)
    return str(filepath)


@pytest.fixture(scope="function")
def test_uneven_h5ad_file(tmp_path: pathlib.Path) -> str:
    # Define a dense matrix with uneven sparsity
    dense = np.array(
        [
            [1, 0, 2, 0, 0, 0],  # 2 non-zeros
            [0, 0, 0, 0, 0, 0],  # empty row
            [0, 3, 0, 4, 0, 0],  # 2 non-zeros
            [5, 0, 0, 0, 0, 6],  # 2 non-zeros
            [0, 7, 8, 0, 0, 0],  # 2 non-zeros
            [0, 0, 0, 0, 9, 0],  # 1 non-zero
            [10, 0, 11, 12, 0, 0],  # 3 non-zeros
            [0, 0, 0, 0, 0, 13],  # 1 non-zero
            [14, 0, 0, 0, 15, 0],  # 2 non-zeros
            [0, 16, 0, 0, 0, 0],  # 1 non-zero
            [0, 16, 0, 0, 0, 0],  # 1 non-zero
        ],
        dtype=np.float32,
    )

    n_obs, n_vars = dense.shape
    X = csr_matrix(dense)

    # Annotate cells and genes
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Write to tmp_path
    filepath = tmp_path / "test_uneven.h5ad"
    adata.write_h5ad(filepath)
    return str(filepath)


def test_load_simple(test_even_h5ad_file: str):
    indices = ann_split_data([test_even_h5ad_file], batch_size=2, test_size=0.0, validation_size=0.0)
    data_module = AnnDataModule(indices=indices, dataset=DistributedAnnDataset, prefetch_factor=2, sparse_keys=["X"])
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    for i, data in enumerate(train_loader):
        data = data_module.on_after_batch_transfer(data, i)
        n, m = data.shape
        assert n > 0
        assert m > 0
        assert isinstance(data, torch.Tensor)
        assert not data.is_sparse
        assert not data.is_sparse_csr


def test_load_with_tuple(test_even_h5ad_file: str):
    indices = ann_split_data([test_even_h5ad_file], batch_size=2, test_size=0.0, validation_size=0.0)

    class DistributedAnnDatasetWithTuple(DistributedAnnDataset):
        def transform(self, start: int, end: int):
            X = super().transform(start, end)
            if X is None:
                return None
            return (X,)

    data_module = AnnDataModule(
        indices=indices, dataset=DistributedAnnDatasetWithTuple, prefetch_factor=2, sparse_keys=["X"]
    )
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    for i, data in enumerate(train_loader):
        data = data_module.on_after_batch_transfer(data, i)
        assert isinstance(data, Iterable)
        assert len(data) == 1
        n, m = data[0].shape
        assert n > 0
        assert m > 0
        assert isinstance(data[0], torch.Tensor)
        assert not data[0].is_sparse
        assert not data[0].is_sparse_csr


def test_load_with_dict(test_even_h5ad_file: str):
    indices = ann_split_data([test_even_h5ad_file], batch_size=2, test_size=0.0, validation_size=0.0)

    class DistributedAnnDatasetWithDict(DistributedAnnDataset):
        def transform(self, start: int, end: int):
            X = super().transform(start, end)
            if X is None:
                return None
            return {"X": X}

    data_module = AnnDataModule(
        indices=indices, dataset=DistributedAnnDatasetWithDict, prefetch_factor=2, sparse_keys=["X"]
    )
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    for i, data in enumerate(train_loader):
        data = data_module.on_after_batch_transfer(data, i)
        assert isinstance(data, dict)
        assert "X" in data
        n, m = data["X"].shape
        assert n > 0
        assert m > 0
        assert isinstance(data["X"], torch.Tensor)
        assert not data["X"].is_sparse
        assert not data["X"].is_sparse_csr


def test_load_uneven(test_uneven_h5ad_file: str):
    indices = ann_split_data([test_uneven_h5ad_file], batch_size=2, test_size=0.0, validation_size=0.0)
    data_module = AnnDataModule(indices=indices, dataset=DistributedAnnDataset, prefetch_factor=2, sparse_keys=["X"])
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    for i, data in enumerate(train_loader):
        data = data_module.on_after_batch_transfer(data, i)
        n, m = data.shape
        assert n > 0
        assert m > 0
        assert isinstance(data, torch.Tensor)
        assert not data.is_sparse
        assert not data.is_sparse_csr


def test_load_multiple_files(test_even_h5ad_file: str, test_uneven_h5ad_file: str):
    indices = ann_split_data(
        [test_even_h5ad_file, test_uneven_h5ad_file], batch_size=2, test_size=0.0, validation_size=0.0
    )
    data_module = AnnDataModule(indices=indices, dataset=DistributedAnnDataset, prefetch_factor=2, sparse_keys=["X"])
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    for i, data in enumerate(train_loader):
        data = data_module.on_after_batch_transfer(data, i)
        n, m = data.shape
        assert n > 0
        assert m > 0
        assert isinstance(data, torch.Tensor)
        assert not data.is_sparse
        assert not data.is_sparse_csr


def test_load_with_callbacks(test_even_h5ad_file: str):
    def before_dense_cb(x, idx):
        # just a dummy callback that adds 1 to all elements
        return x * 1

    def after_dense_cb(x, idx):
        return x / (x.max() + 1)

    indices = ann_split_data([test_even_h5ad_file], batch_size=2, test_size=0.0, validation_size=0.0)
    data_module = AnnDataModule(
        indices=indices,
        dataset=DistributedAnnDataset,
        prefetch_factor=2,
        sparse_keys=["X"],
        before_dense_cb=before_dense_cb,
        after_dense_cb=after_dense_cb,
    )
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    for i, data in enumerate(train_loader):
        data = data_module.on_after_batch_transfer(data, i)
        n, m = data.shape
        assert n > 0
        assert m > 0
        assert isinstance(data, torch.Tensor)
        assert not data.is_sparse
        assert not data.is_sparse_csr
        assert torch.all(data < 1)


def test_custom_dataset(test_even_h5ad_file: str):
    from protoplast.scrna.anndata.torch_dataloader import DistributedCellLineAnnDataset

    indices = ann_split_data(
        [test_even_h5ad_file], batch_size=2, test_size=0.0, validation_size=0.0, metadata_cb=cell_line_metadata_cb
    )
    data_module = AnnDataModule(
        indices=indices,
        dataset=DistributedCellLineAnnDataset,
        prefetch_factor=2,
        sparse_keys=["X"],
    )
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    for i, data in enumerate(train_loader):
        data = data_module.on_after_batch_transfer(data, i)
        assert isinstance(data, Iterable)
        assert len(data) == 2
        n, m = data[0].shape
        assert n > 0
        assert m > 0
        assert isinstance(data[0], torch.Tensor)
        assert not data[0].is_sparse
        assert not data[0].is_sparse_csr
        assert isinstance(data[1], torch.Tensor)
        assert data[1].dtype == torch.int64
        assert data[1].shape[0] == n

def test_block_based_dataset(test_even_h5ad_file: str):
    data_module = AnnDataModule(dataset=BlockBasedAnnDataset, 
                prefetch_factor=2, sparse_keys=["X"], file_paths=[test_even_h5ad_file],
                            ds_batch_size=2,
                            block_size=1,
                            load_factor=2
                        )
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    for i, data in enumerate(train_loader):
        X, cell_idx = data
        X = data_module.on_after_batch_transfer(X, i)
        n, m = X.shape
        assert n == 2 # each batch has 2 cells
        assert m == 5 # each cell has 5 features
        assert isinstance(X, torch.Tensor)
        assert not X.is_sparse
        assert not X.is_sparse_csr
        assert isinstance(cell_idx, torch.Tensor)
        assert cell_idx.shape[0] == n
        assert cell_idx.dtype == torch.int64
        assert cell_idx.shape[1] == 2
        
        
def test_process_sparse2(tmpdir: pathlib.Path):
    for i in range(10):
        # Create a random dense matrix
        dense_matrix = np.random.randint(0, 10, size=(200, 200))

        # Convert to CSR sparse matrix
        sparse_matrix = csr_matrix(dense_matrix)
        
        # Create as anndata object
        adata = ad.AnnData(X=sparse_matrix)
        
        filepath = str(tmpdir / "test.h5ad")
        adata.write_h5ad(filepath)

        data_module = AnnDataModule(dataset=BlockBasedAnnDataset, prefetch_factor=2, sparse_keys=["X"], file_paths=[filepath],
                                    ds_batch_size=2,
                                    block_size=2,
                                    load_factor=2
        )
        data_module.setup(stage="fit")
        
        start = np.random.randint(1, 50)
        end = np.random.randint(150, 200)
        subset_sparse = data_module.train_ds._process_sparse2(adata.X, start, end)
        
        # Verify that this is correct
        assert np.array_equal(subset_sparse.todense(), adata.X[start : end].todense())