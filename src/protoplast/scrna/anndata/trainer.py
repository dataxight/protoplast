#   Copyright 2025 DataXight, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import os
import time
import warnings
from collections.abc import Callable

import anndata
import lightning.pytorch as pl
import ray
import ray.train
import ray.train.lightning
import ray.train.torch
import torch
from beartype import beartype
from lightning.pytorch.strategies import Strategy

from .strategy import SequentialShuffleStrategy, ShuffleStrategy
from .torch_dataloader import AnnDataModule, DistributedAnnDataset, cell_line_metadata_cb


class RayTrainRunner:
    """A class to initialize the training this class automatically initializes Ray cluster or
    detect whether an existing cluster exist if there is an existing cluster it will automatically
    connect to it refer to `ray.init()` behavior

    Parameters
    ----------
    Model : type[pl.LightningModule]
        PyTorch Lightning model class
    Ds : type[DistributedAnnDataset]
        DistributedAnnDataset class
    model_keys : list[str]
        Keys to pass to model from `metadata_cb`
    metadata_cb : Callable[[anndata.AnnData, dict], None], optional
        Callback to mutate metadata recommended for passing data from `obs` or `var`
        or any additional data your models required
        by default cell_line_metadata_cb
    before_dense_cb : Callable[[torch.Tensor, str  |  int], torch.Tensor], optional
        Callback to perform before densification of sparse matrix where the data at this point
        is still a sparse CSR Tensor, by default None
    after_dense_cb : Callable[[torch.Tensor, str  |  int], torch.Tensor], optional
        Callback to perform after densification of sparse matrix where the data at this point
        is a dense Tensor, by default None
    shuffle_strategy : ShuffleStrategy, optional
        Strategy to split or randomize the data during the training, by default SequentialShuffleStrategy
    runtime_env_config : dict | None, optional
        These env config is to pass the RayTrainer processes, by default None
    address : str | None, optional
        Override ray address, by default None
    ray_trainer_strategy : Strategy | None, optional
        Override Ray Trainer Strategy if this is None it will default to RayDDP, by default None
    sparse_key : str, optional
        _description_, by default "X",
    Returns
    -------
    RayTrainRunner
        Use this class to start the training

    """

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
        sparse_key: str = "X",
    ):
        self.Model = Model
        self.Ds = Ds
        self.model_keys = model_keys
        self.metadata_cb = metadata_cb
        self.shuffle_strategy = shuffle_strategy
        self.sparse_key = sparse_key
        self.before_dense_cb = before_dense_cb
        self.after_dense_cb = after_dense_cb
        if not ray_trainer_strategy:
            self.ray_trainer_strategy = ray.train.lightning.RayDDPStrategy()
        else:
            self.ray_trainer_strategy = ray_trainer_strategy

        # Init ray cluster
        DEFAULT_RUNTIME_ENV_CONFIG = {
            "working_dir": os.getenv("PWD"),  # Allow ray workers to inherit venv at $PWD if there is any
        }
        if runtime_env_config is None:
            runtime_env_config = DEFAULT_RUNTIME_ENV_CONFIG
        ray.init(
            address=address, runtime_env={**DEFAULT_RUNTIME_ENV_CONFIG, **runtime_env_config}, ignore_reinit_error=True
        )

        self.resources = ray.cluster_resources()

    @beartype
    def train(
        self,
        file_paths: list[str],
        batch_size: int = 2000,
        test_size: float = 0.0,
        val_size: float = 0.2,
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
        is_shuffled: bool = False,
        **kwargs,
    ):
        """Start the training

        Parameters
        ----------
        file_paths : list[str]
            List of h5ad AnnData files
        batch_size : int, optional
            How much data to fetch from disk, by default to 2000
        test_size : float, optional
            Fraction of test data for example 0.1 means 10% will be split for testing
            default to 0.0
        val_size : float, optional
            Fraction of validation data for example 0.2 means 20% will be split for validation,
            default to 0.2
        prefetch_factor : int, optional
            Total data fetch is prefetch_factor * batch_size, by default 4
        max_epochs : int, optional
            How many epoch(s) to train with, by default 1
        thread_per_worker : int | None, optional
            Amount of worker for each dataloader, by default None
        num_workers : int | None, optional
            Override number of Ray processes default to number of GPU(s) in the cluster, by default None
        result_storage_path : str, optional
            Path to store the loss, validation and checkpoint, by default "~/protoplast_results"
        ckpt_path : str | None, optional
            Path of the checkpoint if this is specified it will train from checkpoint otherwise it will start the
            training from scratch, by default None
        is_gpu : bool, optional
            By default True turn this off if your system don't have any GPU, by default True
        random_seed : int | None, optional
            Set this to None for real training but for benchmarking and result replication
            you can adjust the seed here, by default 42
        resource_per_worker : dict | None, optional
            This get pass to Ray you can specify how much CPU or GPU each Ray process get, by default None
        Returns
        -------
        Result
            The training result from RayTrainer
        """
        self.result_storage_path = result_storage_path
        self.prefetch_factor = prefetch_factor
        self.max_epochs = max_epochs
        self.kwargs = kwargs
        if not resource_per_worker:
            if not thread_per_worker:
                print("Setting thread_per_worker to half of the available CPUs capped at 4")
                thread_per_worker = min(int((self.resources.get("CPU", 2) - 1) / 2), 4)
            resource_per_worker = {"CPU": thread_per_worker}
        if is_gpu and self.resources.get("GPU", 0) == 0:
            warnings.warn("`is_gpu = True` but there is no GPU found. Fallback to CPU.", UserWarning, stacklevel=2)
            is_gpu = False
        if is_gpu:
            if num_workers is None:
                num_workers = int(self.resources.get("GPU"))
            scaling_config = ray.train.ScalingConfig(
                num_workers=num_workers, use_gpu=True, resources_per_worker=resource_per_worker
            )
        else:
            if num_workers is None:
                num_workers = max(int((self.resources.get("CPU", 2) - 1) / thread_per_worker), 1)
            scaling_config = ray.train.ScalingConfig(
                num_workers=num_workers, use_gpu=False, resources_per_worker=resource_per_worker
            )
        print(f"Using {num_workers} workers with {resource_per_worker} each")
        start = time.time()
        shuffle_strategy = self.shuffle_strategy(
            file_paths,
            batch_size,
            num_workers * thread_per_worker,
            test_size,
            val_size,
            random_seed,
            metadata_cb=self.metadata_cb,
            is_shuffled=is_shuffled,
            **kwargs,
        )
        kwargs.pop("drop_last", None)
        kwargs.pop("pre_fetch_then_batch", None)
        indices = shuffle_strategy.split()
        print(f"Data splitting time: {time.time() - start:.2f} seconds")
        train_config = {"indices": indices, "ckpt_path": ckpt_path, "shuffle_strategy": shuffle_strategy}
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
            shuffle_strategy = config.get("shuffle_strategy")
            ann_dm = AnnDataModule(
                indices,
                Ds,
                self.prefetch_factor,
                self.sparse_key,
                shuffle_strategy,
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
                enable_checkpointing=True,
            )
            trainer = ray.train.lightning.prepare_trainer(trainer)
            trainer.fit(model, datamodule=ann_dm, ckpt_path=ckpt_path)

        return anndata_train_func
