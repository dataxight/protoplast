import os
from collections import Counter
from collections.abc import Callable

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.distributed as td
from torch.utils.data import DataLoader, get_worker_info

import anndata
from protoplast.patches.anndata_read_h5ad_backed import apply_read_h5ad_backed_patch
from protoplast.patches.anndata_remote import apply_file_backing_patch

from .strategy import ShuffleStrategy, SplitInfo

apply_file_backing_patch()
apply_read_h5ad_backed_patch()


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
        max_open_files: int = 3,
    ):
        # use first file as reference first
        self.files = file_paths
        self.sparse_keys = sparse_keys
        # map each gene to an index
        for k, v in metadata.items():
            setattr(self, k, v)
        self.metadata = metadata
        self.batches = indices

        self.fptr = dict()
        self.sample_ptr = Counter()
        self.buf_ptr = Counter()
        self.fptr_buf = dict()
        self.file_idx = {f: i for i, f in enumerate(self.files)}
        self.current_files = set()
        self.current_fp_idx = -1
        self.max_open_files = max_open_files

    @classmethod
    def create_distributed_ds(
        cls, indices: SplitInfo, sparse_keys: list[str], mode: str = "train", max_open_files: int = 3
    ):
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
        indices = indices.to_dict() if isinstance(indices, SplitInfo) else indices
        return cls(
            indices["files"],
            indices[f"{mode}_indices"],
            indices["metadata"],
            sparse_keys,
            max_open_files=max_open_files,
        )

    def _init_rank(self):
        worker_info = get_worker_info()
        if worker_info is None:
            self.wid = 0
            self.nworkers = 1
        else:
            self.wid = worker_info.id
            self.nworkers = worker_info.num_workers
        try:
            w_rank = td.get_rank()
            w_size = td.get_world_size()
        except ValueError:
            w_rank = -1
            w_size = -1
        if w_rank >= 0:
            self.ray_rank = w_rank
            self.ray_size = w_size
        else:
            self.ray_rank = 0
            self.ray_size = 1
        self.global_rank = self.ray_rank * self.nworkers + self.wid
        self.total_workers = self.ray_size * self.nworkers

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
        if mats[0].shape[0] == 0:
            return None
        return tuple(mats)

    @staticmethod
    def _safe_index(obj, idx):
        if isinstance(obj, (pd.DataFrame | pd.Series)):
            return obj.iloc[idx]
        else:
            return obj[idx]

    def _get_data(self, idx, data):
        if (type(data) is list) or (type(data) is tuple):
            yield tuple(self._safe_index(d, idx) for d in data)
        elif isinstance(data, dict):
            yield {k: self._safe_index(v, idx) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            yield data[idx]
        else:
            raise ValueError("Unsupported data type")

    def __len__(self):
        return sum(end - start for i in range(len(self.files)) for start, end in self.batches[i])

    def _init_buffer(self, f):
        start, end = self.batches[self.file_idx[f]][self.buf_ptr[f]]
        self.ad = self.fptr[f]
        # can add code to support multiple buffer if require more randomness and shard it
        # during each iteration but for Tahoe this is good enough
        self.fptr_buf[f] = self.transform(start, end)

    def _get_batch_size(self, f):
        start, end = self.batches[self.file_idx[f]][self.buf_ptr[f]]
        return end - start

    def __iter__(self):
        self._init_rank()
        for i, f in enumerate(self.files):
            if i < self.max_open_files:
                self.fptr[f] = anndata.read_h5ad(f, backed="r")
                self.current_files.add(f)
                self.current_fp_idx = i
                self._init_buffer(f)
        while len(self.current_files) > 0:
            for f in list(self.current_files):
                if (self.sample_ptr[f] % self.total_workers) == self.global_rank:
                    yield from self._get_data(self.sample_ptr[f], self.fptr_buf[f])
                self.sample_ptr[f] += 1
                if self.sample_ptr[f] >= self._get_batch_size(f):
                    if self.buf_ptr[f] >= len(self.batches[self.file_idx[f]]) - 1:
                        # removing current file
                        del self.fptr[f]
                        del self.fptr_buf[f]
                        del self.sample_ptr[f]
                        del self.buf_ptr[f]
                        self.current_files.remove(f)
                        # replacing with new file if exist
                        if self.current_fp_idx < len(self.files):
                            new_file = self.files[self.current_fp_idx]
                            self.fptr[new_file] = anndata.read_h5ad(new_file, backed="r")
                            self.current_files.add(new_file)
                            self.current_fp_idx += 1
                            self._init_buffer(new_file)
                        break
                    self.sample_ptr[f] = 0
                    self.buf_ptr[f] += 1
                    self._init_buffer(f)


class DistributedCellLineAnnDataset(DistributedAnnDataset):
    """
    Example of how to extend DistributedAnnDataset to adapt it for cell line linear
    classification model here self.cell_lines is available through writing the
    metadata_cb correctly
    """

    def transform(self, start: int, end: int):
        X = super().transform(start, end)
        if X is None:
            return None
        line_ids = self.ad.obs["cell_line"].iloc[start:end]
        line_idx = np.searchsorted(self.cell_lines, line_ids)
        return X, torch.tensor(line_idx)


class AnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        indices: dict,
        dataset: DistributedAnnDataset,
        prefetch_factor: int,
        sparse_keys: list[str],
        shuffle_stragey: ShuffleStrategy,
        before_dense_cb: Callable[[torch.Tensor, str | int], torch.Tensor] = None,
        after_dense_cb: Callable[[torch.Tensor, str | int], torch.Tensor] = None,
        max_open_files: int = 3,
    ):
        super().__init__()
        self.indices = indices
        self.dataset = dataset
        num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
        self.loader_config = dict(
            batch_size=shuffle_stragey.mini_batch_size,
            num_workers=num_threads,
            prefetch_factor=prefetch_factor,
            collate_fn=shuffle_stragey.mixer,
            persistent_workers=True,
            drop_last=True,
        )
        self.sparse_keys = sparse_keys
        self.before_dense_cb = before_dense_cb
        self.after_dense_cb = after_dense_cb
        self.max_open_files = max_open_files

    def setup(self, stage):
        # this is not necessary but it is here in case we want to download data to local node in the future
        if stage == "fit":
            self.train_ds = self.dataset.create_distributed_ds(
                self.indices, self.sparse_keys, max_open_files=self.max_open_files
            )
            self.val_ds = self.dataset.create_distributed_ds(
                self.indices, self.sparse_keys, "val", max_open_files=self.max_open_files
            )
        if stage == "test":
            self.val_ds = self.dataset.create_distributed_ds(
                self.indices, self.sparse_keys, "test", max_open_files=self.max_open_files
            )
        if stage == "predict":
            self.predict_ds = self.dataset.create_distributed_ds(
                self.indices, self.sparse_keys, max_open_files=self.max_open_files
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.loader_config)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.loader_config)

    def test_dataloader(self):
        # for now not support testing for splitting will support it soon in the future
        return DataLoader(self.val_ds, **self.loader_config)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, **self.loader_config)

    def densify(self, x, idx: str | int = None):
        if isinstance(x, torch.Tensor):
            if self.before_dense_cb:
                x = self.before_dense_cb(x, idx)
            if x.is_sparse or x.is_sparse_csr:
                x = x.to_dense()
            if self.after_dense_cb:
                x = self.after_dense_cb(x, idx)
        return x

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if (type(batch) is list) or (type(batch) is tuple):
            return [self.densify(d, i) for i, d in enumerate(batch)]
        elif isinstance(batch, dict):
            return {k: self.densify(v, k) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return self.densify(batch)
        else:
            return batch
