import os
from collections.abc import Callable

import lightning.pytorch as pl
import ray
import ray.train
import ray.train.lightning
import ray.train.torch
from torch.utils.data import DataLoader

import anndata

from .lightning_models import BaseAnnDataLightningModule
from .torch_dataloader import DistributedAnnDataset, ann_split_data, cell_line_metadata_cb


class RayTrainRunner:
    def __init__(
        self,
        Model: BaseAnnDataLightningModule,
        Ds: DistributedAnnDataset,
        model_keys: list[str],
        metadata_cb: Callable[[anndata.AnnData, dict], None] = cell_line_metadata_cb,
        splitter: Callable[[str, int, float, Callable[[anndata.AnnData, dict], None]], dict] = ann_split_data,
    ):
        self.Model = Model
        self.Ds = Ds
        self.model_keys = model_keys
        self.metadata_cb = metadata_cb
        self.splitter = splitter

    def train(
        self,
        file_paths: list[str],
        thread_per_worker: int,
        batch_size: int,
        test_size: int,
        prefetch_factor: int = 4,
        max_epochs: int = 1,
        num_workers: int | None = None,
    ):
        self.prefetch_factor = prefetch_factor
        self.max_epochs = max_epochs
        ray.init()
        resources = ray.cluster_resources()
        if resources.get("GPU", 0) <= 0:
            raise Exception("Only support with GPU is available only")
        indices = self.splitter(file_paths, batch_size, test_size, metadata_cb=self.metadata_cb)
        train_config = {
            "batch_size": batch_size,
            "test_size": test_size,
            "indices": indices,
        }
        if num_workers is None:
            num_workers = int(resources.get("GPU"))
        scaling_config = ray.train.ScalingConfig(
            num_workers=num_workers, use_gpu=True, resources_per_worker={"CPU": thread_per_worker}
        )
        my_train_func = self._trainer()
        par_trainer = ray.train.torch.TorchTrainer(
            my_train_func, scaling_config=scaling_config, train_loop_config=train_config
        )
        print("Spawning Ray worker and initiating distributed training")
        result = par_trainer.fit()
        print(result.metrics)

    def _trainer(self):
        Model, Ds, model_keys = self.Model, self.Ds, self.model_keys

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
            loader_config = dict(batch_size=None, num_workers=num_threads, prefetch_factor=self.prefetch_factor)
            train_dl, test_dl = DataLoader(train_ds, **loader_config), DataLoader(test_ds, **loader_config)
            model_params = train_ds.metadata
            if model_keys:
                model_params = {k: v for k, v in train_ds.metadata.items() if k in model_keys}
            model = Model(**model_params)
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
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
