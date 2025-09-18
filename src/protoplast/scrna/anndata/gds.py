import json
import os
from collections.abc import Callable

import torch
import torch.distributed as td
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

import anndata

from .strategy import ShuffleStrategy
from .torch_dataloader import DistributedAnnDataset


def save_to_gds(
    files: list[str],
    shuffle_strategy: type[ShuffleStrategy],
    Ds: type[DistributedAnnDataset],
    output_path: str,
    batch_size: int = 1000,
    metadata_cb: Callable[[anndata.AnnData, dict], None] | None = None,
    sparse_key="X",
):
    os.makedirs(output_path, exist_ok=True)
    strat = shuffle_strategy(files, batch_size, 1, 0.0, 0.0, metadata_cb=metadata_cb)
    indices = strat.split()
    ds = Ds(files, indices.train_indices, indices.metadata, sparse_key, mini_batch_size=batch_size)
    dataloader = DataLoader(ds, batch_size=None, num_workers=min(os.cpu_count(), 10))
    gds_file_path = os.path.join(output_path, "data.gds")
    metadata_file_path = os.path.join(output_path, "metadata.json")
    file = torch.cuda.gds.GdsFile(gds_file_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC)
    metadata = dict(batch_size=batch_size, n_cols=next(iter(dataloader))[0].shape[1], metadata=indices.metadata)
    # structure detection
    first_batch = next(iter(dataloader))

    def to_type_str(d: torch.Tensor):
        if d.is_sparse_csr:
            return "csr"
        else:
            return str(d.dtype)

    if isinstance(first_batch, dict):
        metadata["structure"] = {k: to_type_str(v) for k, v in first_batch.items()}
    elif isinstance(first_batch, (list | tuple)):
        metadata["structure"] = [to_type_str(d) for d in first_batch]
    else:
        raise Exception("Tensor structure is not supported")

    def save_csr_mat(x, ptr):
        crow_indices = x.crow_indices().to(torch.int32).contiguous().cuda()  # shape [n_rows + 1]
        col_indices = x.col_indices().to(torch.int32).contiguous().cuda()  # shape [nnz]
        values = x.values().to(torch.float16).contiguous().cuda()
        offset = (ptr, len(col_indices), x.shape[0])
        file.save_storage(crow_indices.untyped_storage(), ptr)
        ptr += crow_indices.nbytes
        file.save_storage(col_indices.untyped_storage(), ptr)
        ptr += col_indices.nbytes
        file.save_storage(values.untyped_storage(), ptr)
        ptr += values.nbytes
        return ptr, offset

    def save_label(y, ptr):
        y = y.contiguous().cuda()
        file.save_storage(y.untyped_storage(), ptr)
        ptr += y.nbytes
        return ptr

    ptr = 0
    offsets = []
    n = 0
    for batch in tqdm(dataloader, desc="Saving to gds storage"):
        if isinstance(metadata["structure"], dict):
            batch = [batch[k] for k in metadata["structure"]]
        for d in batch:
            # support only CSR should support COO also
            if d.is_sparse_csr:
                ptr, offset = save_csr_mat(d, ptr)
                offsets.append(offset)
            else:
                ptr = save_label(d, ptr)
        n += 1
    metadata["offsets"] = offsets
    metadata["n"] = n
    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f)


class DistributedGdsDataset(torch.utils.data.IterableDataset):
    def __init__(self, gds_dir: str, offsets: list):
        super().__init__()
        metadata_path = os.path.join(gds_dir, "metadata.json")
        gds_path = os.path.join(gds_dir, "data.gds")
        self.gds_file = torch.cuda.gds.GdsFile(gds_path, os.O_RDONLY)
        self.offsets = offsets
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        # to get len working
        self._init_rank()

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
        return len(self.offsets)

    def load_csr_matrix(self, offset, ptr):
        n_cols = self.metadata["n_cols"]
        _, nnz, n_rows = offset
        crow_indices = torch.empty(n_rows + 1, dtype=torch.int32, device="cuda")
        self.gds_file.load_storage(crow_indices.untyped_storage(), ptr)
        ptr += crow_indices.nbytes
        col_indices = torch.empty(nnz, dtype=torch.int32, device="cuda")
        self.gds_file.load_storage(col_indices.untyped_storage(), ptr)
        ptr += col_indices.nbytes
        values = torch.empty(nnz, dtype=torch.float16, device="cuda")
        self.gds_file.load_storage(values.untyped_storage(), ptr)
        ptr += values.nbytes
        return torch.sparse_csr_tensor(
            crow_indices, col_indices, values.to(torch.float32), size=(n_rows, n_cols), device="cuda"
        ), ptr

    def load_label(self, offset, ptr, t: str):
        _, _, n_rows = offset
        y = torch.empty(n_rows, dtype=eval(t), device="cuda")
        self.gds_file.load_storage(y.untyped_storage(), ptr)
        ptr += y.nbytes
        return y, ptr

    def load_data(self, offset):
        structure = self.metadata["structure"]
        ptr, _, _ = offset
        if isinstance(structure, list | tuple):
            d = []
            for t in structure:
                if t == "csr":
                    mat, ptr = self.load_csr_matrix(offset, ptr)
                else:
                    mat, ptr = self.load_label(offset, ptr, t)
                d.append(mat)
        elif isinstance(structure, dict):
            d = {}
            for k, v in structure.items():
                if v == "csr":
                    mat, ptr = self.load_csr_matrix(offset, ptr)
                else:
                    mat, ptr = self.load_label(offset, ptr, v)
                d[k] = mat
        return d

    def __iter__(self):
        self._init_rank()
        for i, offset in enumerate(self.offsets):
            if (i % self.total_workers) == self.global_rank:
                yield self.load_data(offset)
