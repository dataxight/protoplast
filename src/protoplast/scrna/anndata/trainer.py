import os
from collections.abc import Callable

import lightning.pytorch as pl
import ray
import ray.train
import ray.train.lightning
import ray.train.torch

import anndata

from .torch_dataloader import AnnDataModule, DistributedAnnDataset, ann_split_data, cell_line_metadata_cb


class RayTrainRunner:
    def __init__(
        self,
        Model: pl.LightningModule,
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
        result_storage_path: str | None = None,
    ):
        self.result_storage_path = result_storage_path
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
            my_train_func,
            scaling_config=scaling_config,
            train_loop_config=train_config,
            run_config=self.result_storage_path,
        )
        print("Spawning Ray worker and initiating distributed training")
        return par_trainer.fit()

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
            model_params = indices["metadata"]
            ann_dm = AnnDataModule(indices, Ds, self.prefetch_factor)
            if model_keys:
                model_params = {k: v for k, v in model_params.items() if k in model_keys}
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
            trainer.fit(model, datamodule=ann_dm)

        return anndata_train_func
