import os
from collections import Counter
from collections.abc import Callable
import json

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
        sparse_key: str,
        mini_batch_size: int = None,
    ):
        # use first file as reference first
        self.files = file_paths
        self.sparse_key = sparse_key
        self.X = None
        self.ad = None
        # map each gene to an index
        for k, v in metadata.items():
            setattr(self, k, v)
        self.metadata = metadata
        self.batches = indices
        self.mini_batch_size = mini_batch_size

    @classmethod
    def create_distributed_ds(cls, indices: SplitInfo, sparse_key: str, mode: str = "train", **kwargs):
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
            sparse_key,
            mini_batch_size=indices.get("mini_batch_size"),
            **kwargs,
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

    def _get_mat_by_range(self, ad: anndata.AnnData, start: int, end: int) -> sp.csr_matrix:
        if "." in self.sparse_key:
            attr, attr_key = self.sparse_key.split(".")
            X = getattr(ad, attr)[attr_key][start:end]
        else:
            X = getattr(ad, self.sparse_key)[start:end]
        return X

    def transform(self, start: int, end: int):
        # by default we just return the matrix
        # sometimes, the h5ad file stores X as the dense matrix,
        # so we have to make sure it is a sparse matrix before returning
        # the batch item
        if self.X is None:
            # we don't have the X upstream, so we have to incurr IO to fetch it
            self.X = self._get_mat_by_range(self.ad, start, end)
        X = self._process_sparse(self.X)
        return X

    def __len__(self):
        return sum(1 for i in range(len(self.files)) for start, end in self.batches[i])

    def __iter__(self):
        self._init_rank()
        gidx = 0
        total_iter = 0
        for fidx, f in enumerate(self.files):
            self.ad = anndata.read_h5ad(f, backed="r")
            for start, end in self.batches[fidx]:
                if not (gidx % self.total_workers) == self.global_rank:
                    gidx += 1
                    continue
                X = self._get_mat_by_range(self.ad, start, end)
                self.X = X
                if self.mini_batch_size is None:
                    # not fetch-then-batch approach, we yield everything
                    yield self.transform(start, end)
                    total_iter += 1
                else:
                    # fetch-then-batch approach
                    for i in range(0, X.shape[0], self.mini_batch_size):
                        # index on the X coordinates
                        b_start, b_end = i, min(i + self.mini_batch_size, X.shape[0])
                        # index on the adata coordinates
                        global_start, global_end = start + i, min(start + i + self.mini_batch_size, end)
                        self.X = X[b_start:b_end]
                        yield self.transform(global_start, global_end)
                        total_iter += 1
                gidx += 1


class DistributedFileSharingAnnDataset(DistributedAnnDataset):
    def __init__(self, file_paths, indices, metadata, sparse_key, max_open_files: int = 3):
        super().__init__(file_paths, indices, metadata, sparse_key)
        self.max_open_files = max_open_files
        self.fptr = dict()
        self.sample_ptr = Counter()
        self.buf_ptr = Counter()
        self.fptr_buf = dict()
        self.file_idx = {f: i for i, f in enumerate(self.files)}
        self.current_files = set()
        self.current_fp_idx = -1

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

    def _init_buffer(self, f):
        start, end = self.batches[self.file_idx[f]][self.buf_ptr[f]]
        self.ad = self.fptr[f]
        # can add code to support multiple buffer if require more randomness and shard it
        # during each iteration but for Tahoe this is good enough
        self.fptr_buf[f] = self.transform(start, end)

    def _get_batch_size(self, f):
        start, end = self.batches[self.file_idx[f]][self.buf_ptr[f]]
        return end - start
    
    def __len__(self):
        return sum(end-start for i in range(len(self.files)) for start, end in self.batches[i])

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
        line_ids = self.ad.obs["cell_line"].iloc[start:end]
        line_idx = np.searchsorted(self.cell_lines, line_ids)
        return X, torch.tensor(line_idx)
    



class AnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        indices: dict,
        dataset: DistributedAnnDataset,
        prefetch_factor: int,
        sparse_key: str,
        shuffle_strategy: ShuffleStrategy,
        before_dense_cb: Callable[[torch.Tensor, str | int], torch.Tensor] = None,
        after_dense_cb: Callable[[torch.Tensor, str | int], torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__()
        self.indices = indices
        self.dataset = dataset
        num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
        self.loader_config = dict(
            num_workers=num_threads,
        )
        if num_threads > 0:
            self.loader_config["prefetch_factor"] = prefetch_factor
            self.loader_config["persistent_workers"] = True
        if shuffle_strategy.is_mixer:
            self.loader_config["batch_size"] = shuffle_strategy.mini_batch_size
            self.loader_config["collate_fn"] = shuffle_strategy.mixer
            self.loader_config["drop_last"] = True
        else:
            self.loader_config["batch_size"] = None
        self.sparse_key = sparse_key
        self.before_dense_cb = before_dense_cb
        self.after_dense_cb = after_dense_cb
        self.kwargs = kwargs

    def setup(self, stage):
        # this is not necessary but it is here in case we want to download data to local node in the future
        if stage == "fit":
            self.train_ds = self.dataset.create_distributed_ds(self.indices, self.sparse_key, **self.kwargs)
            self.val_ds = self.dataset.create_distributed_ds(self.indices, self.sparse_key, "val", **self.kwargs)
        if stage == "test":
            self.val_ds = self.dataset.create_distributed_ds(self.indices, self.sparse_key, "test", **self.kwargs)
        if stage == "predict":
            self.predict_ds = self.dataset.create_distributed_ds(self.indices, self.sparse_key, **self.kwargs)

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
                if x.shape[0] <= 256 and not self.after_dense_cb:
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
