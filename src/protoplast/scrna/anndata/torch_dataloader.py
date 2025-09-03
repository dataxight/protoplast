import os
import random
import warnings
from collections.abc import Callable

import lightning.pytorch as pl
import numpy as np
import scipy.sparse as sp
import torch
import torch.distributed as td
from torch.utils.data import DataLoader, get_worker_info

import anndata
from protoplast.patches.anndata_read_h5ad_backed import apply_read_h5ad_backed_patch
from protoplast.patches.anndata_remote import apply_file_backing_patch

apply_file_backing_patch()
apply_read_h5ad_backed_patch()


def ann_split_data(
    file_paths: list[str],
    batch_size: int,
    test_size: float | None = None,
    validation_size: float | None = None,
    random_seed: int | None = 42,
    metadata_cb: Callable[[anndata.AnnData, dict], None] | None = None,
):
    def to_batches(n):
        return [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    rng = random.Random(random_seed) if random_seed else random.Random()

    # First pass: compute total batches across all files
    file_batches = []
    total_batches = 0
    metadata = dict()
    for i, fp in enumerate(file_paths):
        ad = anndata.read_h5ad(fp, backed="r")
        if i == 0 and metadata_cb:
            metadata_cb(ad, metadata)

        n_obs = ad.n_obs
        if batch_size > n_obs:
            warnings.warn(
                f"Batch size ({batch_size}) is greater than number of observations "
                f"in file {fp} ({n_obs}). Only one batch will be created.",
                stacklevel=2,
            )

        batches = to_batches(n_obs)
        total_batches += len(batches)
        file_batches.append(batches)

    # Safety check
    if (test_size or 0) + (validation_size or 0) > 1:
        raise ValueError("test_size + validation_size must be <= 1")

    # How many batches should go to validation & test globally?
    val_total = int(total_batches * validation_size) if validation_size else 0
    test_total = int(total_batches * test_size) if test_size else 0

    train_datas, validation_datas, test_datas = [], [], []

    # Second pass: allocate splits proportionally per file
    for batches in file_batches:
        rng.shuffle(batches)
        n = len(batches)

        val_n = int(round(n / total_batches * val_total)) if validation_size else 0
        test_n = int(round(n / total_batches * test_total)) if test_size else 0

        val_split = batches[:val_n]
        test_split = batches[val_n : val_n + test_n]
        train_split = batches[val_n + test_n :]

        validation_datas.append(val_split)
        test_datas.append(test_split)
        train_datas.append(train_split)

    return dict(
        files=file_paths,
        train_indices=train_datas,
        val_indices=validation_datas,
        test_indices=test_datas,
        metadata=metadata,
    )


def cell_line_metadata_cb(ad: anndata.AnnData, metadata: dict):
    """
    Example callback for adding cell line metadata when you use this
    with DistributedAnnDataset it will automatically assign all of the key
    and values to an instance variable
    for example metadata["cell_lines"] will be avaliable as self.cell_lines
    in DistributedAnnDataset where you can use it for transformation
    """
    metadata["cell_lines"] = ad.obs["cell_line"].cat.categories.to_list()
    metadata["num_genes"] = ad.var.shape[0]
    metadata["num_classes"] = len(metadata["cell_lines"])


class DistributedAnnDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_paths: list[str],
        indices: list[list[int]],
        metadata: dict,
        sparse_keys: list[str],
        transform_type: str = "tuple",
    ):
        # use first file as reference first
        self.files = file_paths
        self.sparse_keys = sparse_keys
        self.transform_type = transform_type
        # map each gene to an index
        for k, v in metadata.items():
            setattr(self, k, v)
        self.metadata = metadata
        worker_info = get_worker_info()
        if worker_info is None:
            self.wid = 0
            self.nworkers = 1
        else:
            self.wid = worker_info.id
            self.nworkers = worker_info.num_workers
        w_rank = td.get_rank()
        w_size = td.get_world_size()
        if w_rank >= 0:
            self.ray_rank = w_rank
            self.ray_size = w_size
        else:
            self.ray_rank = 0
            self.ray_size = 1
        self.batches = indices

    @classmethod
    def create_distributed_ds(cls, indices: dict, sparse_keys: list[str], mode: str = "train"):
        """
        indices is in the following format
        {
            "files": [path to anndata must correspond to indices],
            "train_indices": [[correspond to files[0]], [correspond to files[i]] ],
            "test_indices": [[correspond to files[0]], [correspond to files[i]] ],
            "metadata": {
                ...,
                depends on metadata_cb read more on cell_line_metadata_cb
            }
        }
        """
        return cls(indices["files"], indices[f"{mode}_indices"], indices["metadata"], sparse_keys)

    def _process_sparse(self, mat) -> torch.Tensor:
        if sp.issparse(mat):
            return torch.sparse_csr_tensor(
                torch.from_numpy(mat.indptr).long(),
                torch.from_numpy(mat.indices).long(),
                torch.from_numpy(mat.data).float(),
                mat.shape,
            )
        return torch.from_numpy(mat).float()

    def transform(self, start: int, end: int):
        mats = []
        for k in self.sparse_keys:
            if "." in k:
                attr, attr_k = k.split(".")
                mat = getattr(self.ad, attr)[attr_k][start:end]
                mats.append(self._process_sparse(mat))
            else:
                mat = getattr(self.ad, k)[start:end]
                mats.append(self._process_sparse(mat))
        if len(mats) == 1:
            return mats[0]
        return tuple(mats)

    def __iter__(self):
        gidx = 0
        for fidx, f in enumerate(self.files):
            self.ad = anndata.read_h5ad(f, backed="r")
            for start, end in self.batches[fidx]:
                if gidx % self.ray_size == self.ray_rank and gidx % self.nworkers == self.wid:
                    yield self.transform(start, end)
                gidx += 1


class DistrbutedCellLineAnnDataset(DistributedAnnDataset):
    """
    Example of how to extend DistributedAnnDataset to adapt it for cell line linear
    classification model here self.cell_lines is available through writing the
    metadata_cb correctly
    """

    def transform(self, start: int, end: int):
        X = super().transform(start, end)
        line_ids = self.ad.obs["cell_line"].iloc[start:end]
        line_idx = np.searchsorted(self.cell_lines, line_ids)
        return X, torch.tensor(line_idx)


class AnnDataModule(pl.LightningDataModule):
    def __init__(self, indices: dict, dataset: DistributedAnnDataset, prefetch_factor: int, sparse_keys: list[str]):
        super().__init__()
        self.indices = indices
        self.dataset = dataset
        num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
        self.loader_config = dict(batch_size=None, num_workers=num_threads, prefetch_factor=prefetch_factor)
        self.sparse_keys = sparse_keys

    def setup(self, stage):
        # this is not necessary but it is here in case we want to download data to local node in the future
        if stage == "fit":
            self.train_ds = self.dataset.create_distributed_ds(self.indices, self.sparse_keys)
            self.val_ds = self.dataset.create_distributed_ds(self.indices, self.sparse_keys, "val")
        if stage == "test":
            self.val_ds = self.dataset.create_distributed_ds(self.indices, self.sparse_keys, "test")
        if stage == "predict":
            self.predict_ds = self.dataset.create_distributed_ds(self.indices, self.sparse_keys)

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.loader_config)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.loader_config)

    def test_dataloader(self):
        # for now not support testing for splitting will support it soon in the future
        return DataLoader(self.val_ds, **self.loader_config)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, **self.loader_config)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        def densify(x):
            if isinstance(x, torch.Tensor):
                if x.is_sparse or getattr(x, "is_sparse_csr", False):
                    return x.to_dense()
            return x

        if isinstance(batch, tuple):
            return tuple([densify(d) for d in batch])
        elif isinstance(batch, dict):
            return {k: densify(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return densify(batch)
        else:
            return batch
