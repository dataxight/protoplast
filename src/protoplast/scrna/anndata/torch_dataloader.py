import os
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
    ):
        # use first file as reference first
        self.files = file_paths
        self.sparse_keys = sparse_keys
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
        self.batches = indices
        if self.ray_size > 1:
            # make sure each worker work on different when yielded
            r = self.ray_rank % len(self.files)
            self.files = self.files[r:] + self.files[:r]
        else:
            r = 0
        self.file_index_map = {
            idx: (idx + r) % len(file_paths)
            for idx in range(len(file_paths))
        }

    @classmethod
    def create_distributed_ds(cls, indices: SplitInfo, sparse_keys: list[str], mode: str = "train"):
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
        if mats[0].shape[0] == 0:
            return None
        return tuple(mats)
    
    def get_data(self, start: int, end: int):
        data = self.transform(start, end)
        if (type(data) is list) or (type(data) is tuple):
            for idx in range(len(data[0])):
                yield tuple(d[idx] for d in data)
        elif isinstance(data, dict):
            for idx in range(len(data[next(iter(data))])):
                yield {k: v[idx] for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            for idx in range(data.shape[0]):
                yield data[idx]
        else:
            raise ValueError("Unsupported data type")
    

    def __len__(self):
        return sum(
            end-start 
            for i in range(len(self.files)) 
            for start,end in self.batches[i] 
        )

    def __iter__(self):
        global_rank = self.ray_rank * self.nworkers + self.wid
        total_workers = self.ray_size * self.nworkers

        for fidx, f in enumerate(self.files):
            orig_idx = self.file_index_map[fidx]
            self.ad = anndata.read_h5ad(f, backed="r")
            for lidx, (start, end) in enumerate(self.batches[orig_idx]):
                if lidx % total_workers == global_rank:
                    yield from self.get_data(start, end)


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
