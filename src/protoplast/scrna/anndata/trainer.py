import os

import lightning.pytorch as pl
import ray
import ray.train
import ray.train.lightning
from torch.utils.data import DataLoader

from .lightning_models import BaseAnnDataLightningModule
from .torch_dataloader import DistributedAnnDataset


def new_trainer(
    Model: BaseAnnDataLightningModule, Ds: DistributedAnnDataset, model_keys: list[str] | None = None, **kwargs
):
    def anndata_train_func(config):
        ctx = ray.train.get_context()
        if ctx:
            rank = ctx.get_world_rank()
        else:
            rank = 0
        indices = config.get("indices")
        num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
        print(f"=========Starting the training on {rank} with num threads: {num_threads}=========")
        train_ds, test_ds = Ds.create_distributed_ds(indices), Ds.create_distributed_ds(indices, is_test=True)
        loader_config = dict(batch_size=None, num_workers=num_threads, prefetch_factor=kwargs.get("prefetch_factor", 4))
        train_dl, test_dl = DataLoader(train_ds, **loader_config), DataLoader(test_ds, **loader_config)
        model_params = train_ds.metadata
        if model_keys:
            model_params = {k: v for k, v in train_ds.metadata.items() if k in model_keys}
        model = Model(**model_params)
        trainer = pl.Trainer(
            max_epochs=kwargs.get("max_epochs", 1),
            devices="auto",
            accelerator="auto",
            strategy=ray.train.lightning.RayDDPStrategy(),
            plugins=[ray.train.lightning.RayLightningEnvironment()],
            callbacks=[ray.train.lightning.RayTrainReportCallback()],
            enable_checkpointing=False,
        )
        trainer = ray.train.lightning.prepare_trainer(trainer)
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)

    return anndata_train_func
