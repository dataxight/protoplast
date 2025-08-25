import torch
from typing import Optional, Callable, List
import anndata
import random
from torch.utils.data import get_worker_info
import torch.distributed as td
import json
import numpy as np

from protoplast.patches.anndata_read_h5ad_backed import apply_read_h5ad_backed_patch
from protoplast.patches.anndata_remote import apply_file_backing_patch

apply_file_backing_patch()
apply_read_h5ad_backed_patch()

def ann_split_data(file_paths: List[str], batch_size: int, test_size: Optional[float] = None, random_seed: Optional[int] = 42, metadata_cb: Optional[Callable[[anndata.AnnData, dict], None]] = None):
    def to_batches(n):
        return [(i, i+batch_size) for i in range(0, n, batch_size)]
    rng = random.Random()
    if random_seed:
        rng = random.Random(random_seed)
    # First pass: compute total batches across all files
    file_batches = []
    total_batches = 0
    metadata = dict()
    for i, fp in enumerate(file_paths):
        ad = anndata.read_h5ad(fp, backed="r")
        if i == 0:
            if metadata_cb:
                metadata_cb(ad, metadata)
        n_obs = ad.n_obs
        batches = to_batches(n_obs)
        total_batches += len(batches)
        file_batches.append(batches)

    # How many batches should go to validation globally?
    val_total = int(total_batches * test_size) if test_size else 0

    train_datas = []
    validation_datas = []

    # Second pass: allocate validation proportionally, per file
    for batches in file_batches:
        rng.shuffle(batches)
        n = len(batches)
        val_n = int(round(n / total_batches * val_total)) if test_size else 0
        val_split = batches[:val_n]
        train_split = batches[val_n:]

        validation_datas.append(val_split)  # grouped by file
        train_datas.append(train_split)     # grouped by file

    return train_datas, validation_datas, metadata


def cell_line_metadata_cb(ad: anndata.AnnData, metadata: dict):
    """
    Example callback for adding cell line metadata when you use this
    with DistributedAnnDataset it will automatically assign all of the key
    and values to an instance variable 
    for example metadata["cell_lines"] will be avaliable as self.cell_lines
    in DistributedAnnDataset where you can use it for transformation
    """
    metadata["cell_lines"] = ad.obs['cell_line'].cat.categories.to_list()
    metadata["num_genes"] = ad.var.shape[0]
    metadata["num_classes"] = len(metadata["cell_lines"])
        

class DistributedAnnDataset(torch.utils.data.IterableDataset):

    def __init__(self, file_paths: list[str], indices: list[list[int]], metadata: dict, is_test: bool = False):
        # use first file as reference first
        self.files = file_paths
        # map each gene to an index
        for k,v in metadata.items():
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
        if w_rank >= 0 and not is_test:
            self.ray_rank = w_rank
            self.ray_size = w_size
        else:
            self.ray_rank = 0
            self.ray_size = 1
        self.batches = indices


    @classmethod
    def create_distributed_ds(cls, indices: dict, is_test: bool = False):
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
        ikey = "train_indices"
        if is_test:
            ikey = "test_indices"
        return cls(indices["files"], indices[ikey], indices["metadata"], is_test=is_test)

    
    def process_X(self, start: int, end: int) -> torch.Tensor:
        sparse = self.sparse[start:end]
        sparse_torch = torch.sparse_csr_tensor(
            torch.from_numpy(sparse.indptr).long(),
            torch.from_numpy(sparse.indices).long(),
            torch.from_numpy(sparse.data).float(),
            sparse.shape
        )
        return sparse_torch
    
    def transform(self, ad: anndata.AnnData, start: int, end: int):
        X = self.process_X(start, end)
        return X


    def __iter__(self):
        gidx = 0
        for fidx, f in enumerate(self.files):
            ad = anndata.read_h5ad(f, backed="r")
            self.sparse = ad.X
            for (start, end) in self.batches[fidx]:
                if gidx % self.ray_size == self.ray_rank and gidx % self.nworkers == self.wid:
                    yield self.transform(ad, start, end)
                gidx += 1


class DistrbutedCellLineAnnDataset(DistributedAnnDataset):
    """
    Example of how to extend DistributedAnnDataset to adapt it for cell line linear
    classification model here self.cell_lines is available through writing the 
    metadata_cb correctly
    """
    def transform(self, ad: anndata.AnnData, start: int, end: int):
        X = super().transform(ad, start, end)
        line_ids = ad.obs["cell_line"].iloc[start:end]
        line_idx = np.searchsorted(self.cell_lines, line_ids)
        return X, torch.tensor(line_idx)
