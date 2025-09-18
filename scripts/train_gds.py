from protoplast.scrna.anndata.torch_dataloader import DistributedCellLineAnnDataset, cell_line_metadata_cb
from protoplast.scrna.anndata.gds import DistributedGdsDataset
from protoplast.scrna.anndata.lightning_models import LinearClassifier
from protoplast.scrna.anndata.strategy import SequentialShuffleStrategy
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import time


def train_h5ad(files: list[str]):
    batch_size = 1000
    strat = SequentialShuffleStrategy(files, batch_size, 1, 0., 0.2, metadata_cb=cell_line_metadata_cb)
    indices = strat.split()
    train_ds = DistributedCellLineAnnDataset(files, indices.train_indices, indices.metadata, "X", mini_batch_size=batch_size)
    train_dataloader = DataLoader(train_ds, batch_size=None, num_workers=10)
    val_ds = DistributedCellLineAnnDataset(files, indices.val_indices, indices.metadata, "X", mini_batch_size=batch_size)
    val_dataloader = DataLoader(val_ds, batch_size=None, num_workers=10)
    model = LinearClassifier(indices.metadata["num_genes"], indices.metadata["num_classes"])
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        enable_checkpointing=False,
    )
    start = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print(trainer.callback_metrics)
    end = time.perf_counter()
    print(f"Elapsed time: {end-start:.6f} seconds")

def train_gsd(gds_dir: str, val_size=0.2):
    # a workaround for now
    ds = DistributedGdsDataset(gds_dir, [])
    # hard code for now we can figure this out later
    n = len(ds.metadata["offsets"])
    n_train = int(n*(1-val_size))
    train_offsets = ds.metadata["offsets"][:n_train]
    val_offsets = ds.metadata["offsets"][n_train:]
    train_ds = DistributedGdsDataset(gds_dir, train_offsets)
    val_ds = DistributedGdsDataset(gds_dir, val_offsets)
    model = LinearClassifier(ds.metadata["metadata"]["num_genes"], ds.metadata["metadata"]["num_classes"])
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        enable_checkpointing=False,
    )
    start = time.perf_counter()
    trainer.fit(model, train_dataloaders=DataLoader(train_ds, batch_size=None), val_dataloaders=DataLoader(val_ds, batch_size=None))
    print(trainer.callback_metrics)
    end = time.perf_counter()
    print(f"Elapsed time: {end-start:.6f} seconds")

if __name__ == "__main__":
    train_gsd("/mnt/ham/dx_data/plate3_gpu")
    # train_h5ad(["/mnt/ham/dx_data/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"])

