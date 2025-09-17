from protoplast.scrna.anndata.torch_dataloader import DistributedAnnDataset, DistributedCellLineAnnDataset, cell_line_metadata_cb
from protoplast.scrna.anndata.strategy import SequentialShuffleStrategy
from torch.utils.data import DataLoader
import torch
import os
import json
from tqdm import tqdm


def save_to_gds(files: list[str], output_path: str, batch_size: int = 1000):
    os.makedirs(output_path, exist_ok=True)
    strat = SequentialShuffleStrategy(files, batch_size, 1, 0., 0., metadata_cb=cell_line_metadata_cb)
    indices = strat.split()
    # example with cell line
    ds = DistributedCellLineAnnDataset(files, indices.train_indices, indices.metadata, ["X"])
    dataloader = DataLoader(ds, batch_size=None, num_workers=10)
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
    for (x,y) in tqdm(dataloader, desc="Saving to gds storage", total=len(dataloader)//batch_size):
        crow_indices = x.crow_indices().to(torch.int32).contiguous().cuda()  # shape [n_rows + 1]
        col_indices = x.col_indices().to(torch.int32).contiguous().cuda()    # shape [nnz]
        values = x.values().to(torch.float16).contiguous().cuda()
        offsets.append((ptr, len(col_indices)))
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
        remainder_row = x.shape[0]
    metadata["offsets"] = offsets
    metadata["n"] = n
    metadata["remainder"] = remainder_row
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f)
    

if __name__ == "__main__":
    save_to_gds(["/mnt/ham/dx_data/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"], "/mnt/ham/dx_data/plate3_gpu")

