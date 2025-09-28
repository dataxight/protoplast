#   Copyright 2025 DataXight, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import os
import pathlib
from collections.abc import Iterable
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader

from protoplast.scrna.anndata.strategy import RandomShuffleStrategy, SequentialShuffleStrategy
from protoplast.scrna.anndata.torch_dataloader import (
    AnnDataModule,
    DistributedAnnDataset,
    DistributedFileSharingAnnDataset,
    cell_line_metadata_cb,
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
    n_vars = 6

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


@pytest.fixture
def test_h5ad_plate(tmp_path: pathlib.Path):
    """
    Factory fixture to create a test h5ad file with uneven data.
    Usage: filepath = test_uneven_h5ad_file(plate_no=2)
    """

    def _make(plate_no: int = 1) -> str:
        # Dense matrix with uneven sparsity
        dense = np.array(
            [
                [1, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 3, 0, 4, 0, 0],
                [5, 0, 0, 0, 0, 6],
                [0, 7, 8, 0, 0, 0],
                [0, 0, 0, 0, 9, 0],
                [10, 0, 11, 12, 0, 0],
                [0, 0, 0, 0, 0, 13],
                [14, 0, 0, 0, 15, 0],
                [0, 16, 0, 0, 0, 0],
                [0, 16, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        n_obs, n_vars = dense.shape
        X = csr_matrix(dense)

        # Annotate cells and genes with specified plate_no
        obs = pd.DataFrame({"plate": [plate_no] * n_obs}, index=[f"cell_{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
        adata = ad.AnnData(X=X, obs=obs, var=var)

        # Write to tmp_path
        filepath = tmp_path / f"test_uneven_plate{plate_no}.h5ad"
        adata.write_h5ad(filepath)
        return str(filepath)

    return _make


@pytest.fixture(scope="module", autouse=True)
def set_env_vars():
    os.environ["OMP_NUM_THREADS"] = "0"


def test_same_iteration_per_ray_worker(test_h5ad_plate):
    total_plates = 6
    paths = [test_h5ad_plate(plate_no=i + 1) for i in range(total_plates)]

    total_ray_worker = 3
    thread_per_worker = 3
    batch_size = 3

    shuffle_strategy = SequentialShuffleStrategy(
        paths,
        batch_size=batch_size,
        total_workers=total_ray_worker,
        test_size=0.0,
        validation_size=0.0,
        pre_fetch_then_batch=1,
    )

    indices = shuffle_strategy.split()

    data_module = AnnDataModule(
        indices=indices,
        dataset=DistributedAnnDataset,
        prefetch_factor=2,
        sparse_key="X",
        shuffle_strategy=shuffle_strategy,
    )
    data_module.setup(stage="fit")
    # simulate each ray worker
    total_iters = []
    total_data = 0
    for r in range(total_ray_worker):
        total_iter = 0
        for wr in range(thread_per_worker):
            with (
                mock.patch.object(dist, "get_rank", return_value=r),
                mock.patch.object(dist, "get_world_size", return_value=total_ray_worker),
                mock.patch("protoplast.scrna.anndata.torch_dataloader.get_worker_info") as mock_info,
            ):
                mock_info.return_value = mock.Mock(
                    id=wr,
                    num_workers=thread_per_worker,
                    seed=1234 + wr,
                )
                train_loader = data_module.train_dataloader()
                for i, data in enumerate(train_loader):
                    data = data_module.on_after_batch_transfer(data, i)
                    n, m = data.shape
                    assert n > 0
                    assert m > 0
                    total_data += n
                    assert isinstance(data, torch.Tensor)
                    assert not data.is_sparse
                    assert not data.is_sparse_csr
                    total_iter += 1
        assert total_iter > 0
        total_iters.append(total_iter)
    assert len(np.unique(total_iters)) == 1
    assert total_iters[0] == 6
    # data was dropped because the batches is not divisble by number of ray worker
    assert total_data == (total_plates * 11) - 12


def test_entropy(test_h5ad_plate):
    file1 = test_h5ad_plate(plate_no=1)
    file2 = test_h5ad_plate(plate_no=2)
    file3 = test_h5ad_plate(plate_no=3)

    paths = [file1, file2, file3]

    batch_size = 3
    mini_batch_size = 3

    shuffle_strategy = RandomShuffleStrategy(
        paths,
        batch_size,
        mini_batch_size=mini_batch_size,
        total_workers=1,
        test_size=0.0,
        validation_size=0.0,
        is_shuffled=True,
    )

    indices = shuffle_strategy.split()

    class BenchmarkDistributedAnnDataset(DistributedFileSharingAnnDataset):
        def transform(self, start: int, end: int):
            X = super().transform(start, end)
            plate = self.ad.obs["plate"].iloc[start:end]
            if X is None:
                return None
            return X, plate

    # Initialize dataset and dataloader
    dataset = BenchmarkDistributedAnnDataset(
        file_paths=paths,
        indices=indices.train_indices,
        metadata=indices.metadata,
        sparse_key="X",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=shuffle_strategy.mini_batch_size,
        collate_fn=shuffle_strategy.mixer,
    )
    total_n = 0
    total_unique = 0
    for batch in dataloader:
        X, plates = batch
        assert isinstance(X, torch.Tensor)
        assert isinstance(plates, list) or isinstance(plates, np.ndarray) or isinstance(plates, torch.Tensor)
        assert X.shape[0] == len(plates)
        if len(np.unique(plates)) > 1:
            total_unique += 1
        total_n += 1
    assert total_unique > total_n / 2


def test_load_simple(test_even_h5ad_file: str):
    strategy = SequentialShuffleStrategy(
        [test_even_h5ad_file], batch_size=2, total_workers=1, test_size=0.0, validation_size=0.0, pre_fetch_then_batch=1
    )
    indices = strategy.split()
    data_module = AnnDataModule(
        indices=indices, dataset=DistributedAnnDataset, prefetch_factor=2, sparse_key="X", shuffle_strategy=strategy
    )
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    total_n = 0
    for i, data in enumerate(train_loader):
        data = data_module.on_after_batch_transfer(data, i)
        n, m = data.shape
        assert n > 0
        assert m > 0
        assert isinstance(data, torch.Tensor)
        assert not data.is_sparse
        assert not data.is_sparse_csr
        total_n += 1
    assert total_n == len(train_loader)


def test_load_with_tuple(test_even_h5ad_file: str):
    strategy = SequentialShuffleStrategy(
        [test_even_h5ad_file], batch_size=2, total_workers=1, test_size=0.0, validation_size=0.0, pre_fetch_then_batch=1
    )
    indices = strategy.split()

    class DistributedAnnDatasetWithTuple(DistributedAnnDataset):
        def transform(self, start: int, end: int):
            X = super().transform(start, end)
            if X is None:
                return None
            return (X,)

    data_module = AnnDataModule(
        indices=indices,
        dataset=DistributedAnnDatasetWithTuple,
        prefetch_factor=2,
        sparse_key="X",
        shuffle_strategy=strategy,
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
    strategy = SequentialShuffleStrategy(
        [test_even_h5ad_file], batch_size=2, total_workers=1, test_size=0.0, validation_size=0.0, pre_fetch_then_batch=1
    )
    indices = strategy.split()

    class DistributedAnnDatasetWithDict(DistributedAnnDataset):
        def transform(self, start: int, end: int):
            X = super().transform(start, end)
            if X is None:
                return None
            return {"X": X}

    data_module = AnnDataModule(
        indices=indices,
        dataset=DistributedAnnDatasetWithDict,
        prefetch_factor=2,
        sparse_key="X",
        shuffle_strategy=strategy,
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
    strategy = SequentialShuffleStrategy(
        [test_uneven_h5ad_file],
        batch_size=2,
        total_workers=1,
        test_size=0.0,
        validation_size=0.0,
        pre_fetch_then_batch=1,
    )
    indices = strategy.split()
    data_module = AnnDataModule(
        indices=indices, dataset=DistributedAnnDataset, prefetch_factor=2, sparse_key="X", shuffle_strategy=strategy
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


def test_load_multiple_files(test_even_h5ad_file: str, test_uneven_h5ad_file: str):
    strategy = SequentialShuffleStrategy(
        [test_even_h5ad_file, test_uneven_h5ad_file],
        batch_size=2,
        total_workers=1,
        test_size=0.0,
        validation_size=0.0,
        pre_fetch_then_batch=1,
    )
    indices = strategy.split()
    data_module = AnnDataModule(
        indices=indices, dataset=DistributedAnnDataset, prefetch_factor=2, sparse_key="X", shuffle_strategy=strategy
    )
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    for i, data in enumerate(train_loader):
        data = data_module.on_after_batch_transfer(data, i)
        n, m = data.shape
        assert n > 0
        assert m == 6
        assert isinstance(data, torch.Tensor)
        assert not data.is_sparse
        assert not data.is_sparse_csr


def test_load_with_callbacks(test_even_h5ad_file: str):
    def before_dense_cb(x, idx):
        # just a dummy callback that adds 1 to all elements
        return x * 1

    def after_dense_cb(x, idx):
        return x / (x.max() + 1)

    strategy = SequentialShuffleStrategy(
        [test_even_h5ad_file], batch_size=2, total_workers=1, test_size=0.0, validation_size=0.0, pre_fetch_then_batch=1
    )
    indices = strategy.split()
    data_module = AnnDataModule(
        indices=indices,
        dataset=DistributedAnnDataset,
        prefetch_factor=2,
        sparse_key="X",
        shuffle_strategy=strategy,
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

    strategy = SequentialShuffleStrategy(
        [test_even_h5ad_file],
        batch_size=2,
        total_workers=1,
        test_size=0.0,
        validation_size=0.0,
        pre_fetch_then_batch=1,
        metadata_cb=cell_line_metadata_cb,
    )
    indices = strategy.split()
    data_module = AnnDataModule(
        indices=indices,
        dataset=DistributedCellLineAnnDataset,
        prefetch_factor=2,
        sparse_key="X",
        shuffle_strategy=strategy,
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
