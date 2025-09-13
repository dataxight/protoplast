import os
import time
from collections.abc import Callable, Iterable

import lightning.pytorch as pl
import ray
import ray.train
import ray.train.lightning
import ray.train.torch
import torch
from beartype import beartype
from lightning.pytorch.strategies import Strategy

import anndata

from .strategy import SequentialShuffleStrategy, ShuffleStrategy
from .torch_dataloader import AnnDataModule, DistributedAnnDataset, cell_line_metadata_cb


class RayTrainRunner:
    @beartype
    def __init__(
        self,
        Model: type[pl.LightningModule],
        Ds: type[DistributedAnnDataset],
        model_keys: list[str],
        metadata_cb: Callable[[anndata.AnnData, dict], None] = cell_line_metadata_cb,
        before_dense_cb: Callable[[torch.Tensor, str | int], torch.Tensor] = None,
        after_dense_cb: Callable[[torch.Tensor, str | int], torch.Tensor] = None,
        shuffle_strategy: ShuffleStrategy = SequentialShuffleStrategy,
        runtime_env_config: dict | None = None,
        address: str | None = None,
        ray_trainer_strategy: Strategy | None = None,
        sparse_keys: Iterable[str] = ("X",),
        max_open_files: int = 3,
    ):
        self.Model = Model
        self.Ds = Ds
        self.model_keys = model_keys
        self.metadata_cb = metadata_cb
        self.shuffle_strategy = shuffle_strategy
        self.sparse_keys = sparse_keys
        self.before_dense_cb = before_dense_cb
        self.after_dense_cb = after_dense_cb
        self.max_open_files = max_open_files
        if not ray_trainer_strategy:
            self.ray_trainer_strategy = ray.train.lightning.RayDDPStrategy()
        else:
            self.ray_trainer_strategy = ray_trainer_strategy
        ray.init(address=address, runtime_env=runtime_env_config, ignore_reinit_error=True)
        self.resources = ray.cluster_resources()
        if self.resources.get("GPU", 0) <= 0:
            raise Exception("Only support with GPU is available only")

    @beartype
    def train(
        self,
        file_paths: list[str],
        batch_size: int,
        test_size: float,
        val_size: float,
        prefetch_factor: int = 4,
        max_epochs: int = 1,
        thread_per_worker: int | None = None,
        num_workers: int | None = None,
        result_storage_path: str = "~/protoplast_results",
        # read more here: https://lightning.ai/docs/pytorch/stable/common/trainer.html#fit
        ckpt_path: str | None = None,
        is_gpu: bool = True,
        random_seed: int | None = 42,
        resource_per_worker: dict | None = None,
        is_shuffled: bool = True,
        **kwargs,
    ):
        self.result_storage_path = result_storage_path
        self.prefetch_factor = prefetch_factor
        self.max_epochs = max_epochs
        self.kwargs = kwargs
        if not resource_per_worker:
            if not thread_per_worker:
                print("Setting thread_per_worker to half of the available CPUs capped at 4")
                thread_per_worker = min(int(self.resources.get("CPU", 1) / 2), 4)
            resource_per_worker = {"CPU": thread_per_worker}
        if is_gpu:
            if num_workers is None:
                num_workers = int(self.resources.get("GPU"))
            scaling_config = ray.train.ScalingConfig(
                num_workers=num_workers, use_gpu=True, resources_per_worker=resource_per_worker
            )
        else:
            if num_workers is None:
                num_workers = max(int(self.resources.get("CPU", 1) / thread_per_worker), 1)
            scaling_config = ray.train.ScalingConfig(
                num_workers=num_workers, use_gpu=False, resources_per_worker=resource_per_worker
            )
        print(f"Using {num_workers} workers with {resource_per_worker} each")
        start = time.time()
        shuffle_stragey = self.shuffle_strategy(
            file_paths,
            batch_size,
            num_workers,
            test_size,
            val_size,
            random_seed,
            metadata_cb=self.metadata_cb,
            is_shuffled=is_shuffled,
            **kwargs,
        )
        indices = shuffle_stragey.split()
        print(f"Data splitting time: {time.time() - start:.2f} seconds")
        train_config = {"indices": indices, "ckpt_path": ckpt_path, "shuffle_stragey": shuffle_stragey}
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
            model_params = indices.metadata
            shuffle_stragey = config.get("shuffle_stragey")
            ann_dm = AnnDataModule(
                indices,
                Ds,
                self.prefetch_factor,
                self.sparse_keys,
                shuffle_stragey,
                self.before_dense_cb,
                self.after_dense_cb,
                **self.kwargs,
            )
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
