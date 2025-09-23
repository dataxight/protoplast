import time

import lightning.pytorch as pl
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
from torch.utils.data import DataLoader

from protoplast.scrna.anndata.gds import DistributedGdsDataset
from protoplast.scrna.anndata.lightning_models import LinearClassifier
from protoplast.scrna.anndata.strategy import SequentialShuffleStrategy
from protoplast.scrna.anndata.torch_dataloader import DistributedCellLineAnnDataset, cell_line_metadata_cb


def train_h5ad(files: list[str]):
    batch_size = 1000
    strat = SequentialShuffleStrategy(files, batch_size, 1, 0.0, 0.2, metadata_cb=cell_line_metadata_cb)
    indices = strat.split()
    train_ds = DistributedCellLineAnnDataset(
        files, indices.train_indices, indices.metadata, "X", mini_batch_size=batch_size
    )
    train_dataloader = DataLoader(train_ds, batch_size=None, num_workers=0)
    profiler = AdvancedProfiler(dirpath="advance_profiler", filename="result", dump_stats=True)
    val_ds = DistributedCellLineAnnDataset(
        files, indices.val_indices, indices.metadata, "X", mini_batch_size=batch_size, profiler=profiler
    )
    val_dataloader = DataLoader(val_ds, batch_size=None, num_workers=0)
    model = LinearClassifier(indices.metadata["num_genes"], indices.metadata["num_classes"])
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        enable_checkpointing=False,
        profiler=profiler
    )
    start = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print(trainer.callback_metrics)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.6f} seconds")


def divisble_array(ar, n):
    remainder = len(ar) % n > 0 
    if remainder > 0:
        return ar[:-remainder]
    return ar


def train_gsd(gds_dir: str, val_size=0.2):
    # a workaround for now
    ds = DistributedGdsDataset(gds_dir, [])
    # hard code for now we can figure this out later
    n = len(ds.metadata["offsets"])
    n_train = int(n * (1 - val_size))
    devices = 2
    train_offsets = divisble_array(ds.metadata["offsets"][:n_train], devices)
    val_offsets = divisble_array(ds.metadata["offsets"][n_train:], devices)
    train_ds = DistributedGdsDataset(gds_dir, train_offsets)
    val_ds = DistributedGdsDataset(gds_dir, val_offsets)
    model = LinearClassifier(ds.metadata["metadata"]["num_genes"], ds.metadata["metadata"]["num_classes"])
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        enable_checkpointing=False,
        devices=devices
    )
    start = time.perf_counter()
    trainer.fit(
        model,
        train_dataloaders=DataLoader(train_ds, batch_size=None),
        val_dataloaders=DataLoader(val_ds, batch_size=None),
    )
    print(trainer.callback_metrics)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.6f} seconds")


if __name__ == "__main__":
    # train_gsd("/ephemeral/gds/all_plates")
    # train_gsd("/mnt/ham/dx_data/plate3_gpu")
    # train_h5ad(["/mnt/ham/dx_data/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"])
    train_h5ad(["/ephemeral/tahoe100/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"])
