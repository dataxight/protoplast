from protoplast.scrna.anndata.torch_dataloader import DistributedAnnDataset, DistributedCellLineAnnDataset, cell_line_metadata_cb, DistributedGdsDataset
from protoplast.scrna.anndata.lightning_models import LinearClassifier
from protoplast.scrna.anndata.strategy import SequentialShuffleStrategy
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import time


def train_h5ad(files: list[str]):
    batch_size = 1000
    strat = SequentialShuffleStrategy(files, batch_size, 1, 0., 0., metadata_cb=cell_line_metadata_cb)
    indices = strat.split()
    # example with cell line
    ds = DistributedCellLineAnnDataset(files, indices.train_indices, indices.metadata, ["X"])
    dataloader = DataLoader(ds, batch_size=None, num_workers=10)
    model = LinearClassifier(indices.metadata["num_genes"], indices.metadata["num_classes"])
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        enable_checkpointing=False,
    )
    start = time.perf_counter()
    trainer.fit(model, train_dataloaders=dataloader)
    print(trainer.callback_metrics)
    end = time.perf_counter()
    print(f"Elapsed time: {end-start:.6f} seconds")

def train_gsd(gds_dir: str):
    # example with cell line
    ds = DistributedGdsDataset(gds_dir)
    dataloader = DataLoader(ds, batch_size=None)
    # hard code for now we can figure this out later
    model = LinearClassifier(ds.metadata["n_cols"], ds.metadata["num_classes"])
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        enable_checkpointing=False,
    )
    start = time.perf_counter()
    trainer.fit(model, train_dataloaders=dataloader)
    print(trainer.callback_metrics)
    end = time.perf_counter()
    print(f"Elapsed time: {end-start:.6f} seconds")

if __name__ == "__main__":
    # train_gsd("/mnt/ham/dx_data/plate3_gpu")
    train_h5ad(["/mnt/ham/dx_data/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"])

