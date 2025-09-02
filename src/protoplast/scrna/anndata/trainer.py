import os
from collections.abc import Callable, Iterable

import lightning.pytorch as pl
import ray
import ray.train
import ray.train.lightning
import ray.train.torch
from beartype import beartype
from lightning.pytorch.strategies import Strategy

import anndata

from .torch_dataloader import AnnDataModule, DistributedAnnDataset, ann_split_data, cell_line_metadata_cb


class RayTrainRunner:
    @beartype
    def __init__(
        self,
        Model: type[pl.LightningModule],
        Ds: type[DistributedAnnDataset],
        model_keys: list[str],
        metadata_cb: Callable[[anndata.AnnData, dict], None] = cell_line_metadata_cb,
        splitter: Callable[
            [list[str], int, float, float, int, Callable[[anndata.AnnData, dict], None]], dict
        ] = ann_split_data,
        runtime_env_config: dict | None = None,
        address: str | None = None,
        ray_trainer_strategy: Strategy | None = None,
        sparse_keys: Iterable[str] = ("X",),
    ):
        self.Model = Model
        self.Ds = Ds
        self.model_keys = model_keys
        self.metadata_cb = metadata_cb
        self.splitter = splitter
        self.sparse_keys = sparse_keys
        if not ray_trainer_strategy:
            self.ray_trainer_strategy = ray.train.lightning.RayDDPStrategy()
        else:
            self.ray_trainer_strategy = ray_trainer_strategy
        ray.init(address=address, runtime_env=runtime_env_config)
        self.resources = ray.cluster_resources()
        if self.resources.get("GPU", 0) <= 0:
            raise Exception("Only support with GPU is available only")

    @beartype
    def train(
        self,
        file_paths: list[str],
        thread_per_worker: int,
        batch_size: int,
        test_size: float,
        val_size: float,
        prefetch_factor: int = 4,
        max_epochs: int = 1,
        num_workers: int | None = None,
        result_storage_path: str = "~/protoplast_results",
        # read more here: https://lightning.ai/docs/pytorch/stable/common/trainer.html#fit
        ckpt_path: str | None = None,
    ):
        self.result_storage_path = result_storage_path
        self.prefetch_factor = prefetch_factor
        self.max_epochs = max_epochs
        indices = self.splitter(file_paths, batch_size, test_size, val_size, metadata_cb=self.metadata_cb)
        train_config = {"indices": indices, "ckpt_path": ckpt_path}
        if num_workers is None:
            num_workers = int(self.resources.get("GPU"))
        scaling_config = ray.train.ScalingConfig(
            num_workers=num_workers, use_gpu=True, resources_per_worker={"CPU": thread_per_worker}
        )
        my_train_func = self._trainer()
        par_trainer = ray.train.torch.TorchTrainer(
            my_train_func,
            scaling_config=scaling_config,
            train_loop_config=train_config,
            run_config=ray.train.RunConfig(storage_path=self.result_storage_path),
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
            ckpt_path = config.get("ckpt_path")
            num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
            print(f"=========Starting the training on {rank} with num threads: {num_threads}=========")
            model_params = indices["metadata"]
            ann_dm = AnnDataModule(indices, Ds, self.prefetch_factor, self.sparse_keys)
            if model_keys:
                model_params = {k: v for k, v in model_params.items() if k in model_keys}
            model = Model(**model_params)
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                devices="auto",
                accelerator="auto",
                strategy=self.ray_trainer_strategy,
                plugins=[ray.train.lightning.RayLightningEnvironment()],
                callbacks=[ray.train.lightning.RayTrainReportCallback()],
                enable_checkpointing=False,
            )
            trainer = ray.train.lightning.prepare_trainer(trainer)
            trainer.fit(model, datamodule=ann_dm, ckpt_path=ckpt_path)

        return anndata_train_func
