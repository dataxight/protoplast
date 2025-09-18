import torch
import os
import json
import torch.distributed as td
from typing import Callable
import anndata
from .torch_dataloader import DistributedAnnDataset

from torch.utils.data import DataLoader, get_worker_info
from .strategy import ShuffleStrategy
from tqdm import tqdm


def save_to_gds(files: list[str], shuffle_strategy: type[ShuffleStrategy], Ds: type[DistributedAnnDataset], output_path: str, batch_size: int = 1000, metadata_cb: Callable[[anndata.AnnData, dict], None] | None = None):
    os.makedirs(output_path, exist_ok=True)
    strat = shuffle_strategy(files, batch_size, 1, 0., 0., metadata_cb=metadata_cb)
    indices = strat.split()
    # example with cell line
    ds = Ds(files, indices.train_indices, indices.metadata, ["X"])
    dataloader = DataLoader(ds, batch_size=None, num_workers=min(os.cpu_count, 10))
    gds_file_path = os.path.join(output_path, "data.gds")
    metadata_file_path = os.path.join(output_path, "metadata.json")
    file = torch.cuda.gds.GdsFile(gds_file_path, os.O_CREAT | os.O_RDWR)
    metadata = dict(
        batch_size=batch_size,
        n_classes=indices.metadata["num_classes"],
        n_cols = next(iter(dataloader))[0].shape[1]
    )
    ptr = 0
    offsets = []
    n = 0
    remainder_row = 0
    for (x,y) in tqdm(dataloader, desc="Saving to gds storage", total=(len(dataloader)//batch_size) + 1):
        crow_indices = x.crow_indices().to(torch.int32).contiguous().cuda()  # shape [n_rows + 1]
        col_indices = x.col_indices().to(torch.int32).contiguous().cuda()    # shape [nnz]
        values = x.values().to(torch.float16).contiguous().cuda()
        offsets.append((ptr, len(col_indices), x.shape[0]))
        file.save_storage(crow_indices.untyped_storage(), ptr)
        ptr += crow_indices.nbytes
        file.save_storage(col_indices.untyped_storage(), ptr)
        ptr += col_indices.nbytes
        file.save_storage(values.untyped_storage(), ptr)
        ptr += values.nbytes
        y = y.to(torch.int32).contiguous().cuda()
        file.save_storage(y.untyped_storage(), ptr)
        ptr += y.nbytes
        n += 1
    metadata["offsets"] = offsets
    metadata["n"] = n
    metadata["remainder"] = remainder_row
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f)


class DistributedGdsDataset(torch.utils.data.IterableDataset):
    def __init__(self, gds_dir: str):
        super().__init__()
        metadata_path = os.path.join(gds_dir, "metadata.json")
        gds_path = os.path.join(gds_dir, "data.gds")
        self.gds_file = torch.cuda.gds.GdsFile(gds_path, os.O_RDONLY)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

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

    def __len__(self):
        return self.metadata["n"]

    def load_data(self, offset):
        n_cols = self.metadata["n_cols"]
        ptr, nnz, n_rows = offset
        crow_indices = torch.empty(n_rows + 1, dtype=torch.int32, device="cuda")
        self.gds_file.load_storage(crow_indices.untyped_storage(), ptr)
        ptr += crow_indices.nbytes
        col_indices = torch.empty(nnz, dtype=torch.int32, device="cuda")
        self.gds_file.load_storage(col_indices.untyped_storage(), ptr)
        ptr += col_indices.nbytes
        values = torch.empty(nnz, dtype=torch.float16, device="cuda")
        self.gds_file.load_storage(values.untyped_storage(), ptr)
        ptr += values.nbytes
        X = torch.sparse_csr_tensor(crow_indices, col_indices, values.to(torch.float32), size=(n_rows, n_cols), device="cuda")
        # TODO: support other parameter in the future
        y = torch.empty(n_rows, dtype=torch.int32, device="cuda")
        self.gds_file.load_storage(y.untyped_storage(), ptr)
        return X,y.to(torch.long)


    def __iter__(self):
        self._init_rank()
        offsets = self.metadata["offsets"]
        for i, offset in enumerate(offsets):
            if (i % self.total_workers) == self.global_rank:
                yield self.load_data(offset)


