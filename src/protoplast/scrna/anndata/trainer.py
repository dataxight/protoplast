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


import logging
import os
import time
import uuid
import warnings
from collections.abc import Callable
from typing import Literal

import anndata
import fsspec
import lightning.pytorch as pl
import ray
import ray.train
import ray.train.lightning
import ray.train.torch
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
from beartype import beartype
from lightning.pytorch.strategies import Strategy

from protoplast.patches.file_handler import get_fsspec
import pandas as pd
from ...utils import setup_console_logging
from .strategy import SequentialShuffleStrategy, ShuffleStrategy
from .torch_dataloader import AnnDataModule, DistributedAnnDataset, cell_line_metadata_cb

logger = logging.getLogger(__name__)
setup_console_logging()

def write_predictions_to_file(predictions, output_path: str, format: Literal["csv", "parquet"]):
    """Write predictions to a file in the specified format.

    Parameters
    ----------
    predictions : torch.Tensor | list | dict
        Predictions to write.
    output_path : str
        Path to the output file.
    format : Literal["csv", "parquet"]
        Format of the output file.
    """
    if isinstance(predictions, torch.Tensor):
        preds_numpy = predictions.cpu().numpy()
    elif isinstance(predictions, list):
        preds_numpy = torch.cat(predictions, dim=0).cpu().numpy()
    elif isinstance(predictions, dict):
        preds_numpy = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                preds_numpy[key] = value.cpu().numpy().tolist()
            elif isinstance(value, list):
                if value and isinstance(value[0], torch.Tensor):
                    preds_numpy[key] = torch.cat(value, dim=0).cpu().numpy().tolist()
            else:
                # Assume value is already in the desired format (e.g., numpy)
                preds_numpy[key] = value
    if format == "csv":
        df = pd.DataFrame(preds_numpy)
        with get_fsspec(output_path, "wb") as f:
            df.to_csv(f, index=False)
    elif format == "parquet":
        df = pd.DataFrame(preds_numpy)
        with get_fsspec(output_path, "wb") as f:
            df.to_parquet(f, index=False)

class PredictionWriterCallback(BasePredictionWriter):
    def __init__(self, output_path: str, format: Literal["csv", "parquet"]):
        super().__init__(write_interval="batch")
        self.output_path = output_path
        self.format = format

    def write_on_batch_end(self, trainer, pl_module, predictions, batch_indices, batch, batch_idx, dataloader_idx):
        write_predictions_to_file(
            predictions,
            os.path.join(self.output_path, f"preds_batch_{batch_idx}.{self.format}"),
            self.format,
        )

class DistributedPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, rank: int, format: Literal["csv", "parquet"]):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.rank = rank
        self.format = format

    def write_on_batch_end(self, trainer, pl_module, predictions, batch_indices, batch, batch_idx, dataloader_idx):
        base_name = f"preds_rank_{self.rank}_batch_{batch_idx}"
        write_predictions_to_file(
            predictions,
            os.path.join(self.output_dir, f"{base_name}.{self.format}"),
            self.format,
        )


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
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        }
        if runtime_env_config is None:
            runtime_env_config = DEFAULT_RUNTIME_ENV_CONFIG
        ray.init(
            address=address, runtime_env={**DEFAULT_RUNTIME_ENV_CONFIG, **runtime_env_config}, ignore_reinit_error=True
        )

        self.resources = ray.cluster_resources()

    def _worker_fn(self):
        warnings.filterwarnings(action="ignore", module="ray", category=DeprecationWarning)
        Model, Ds, model_keys = self.Model, self.Ds, self.model_keys

        def worker_fn(config):
            ctx = ray.train.get_context()
            if ctx:
                rank = ctx.get_world_rank()
            else:
                rank = 0
            indices = config.get("indices")
            ckpt_path = config.get("ckpt_path")
            scratch_path = config.get("scratch_path")
            scratch_content = config.get("scratch_content")
            logger.debug("Verifying storage path on worker node")
            try:
                file = get_fsspec(scratch_path, "r")
                read_content = file.read()
                file.close()
            except Exception as e:
                logger.error("Failed to access shared storage path: %s", scratch_path, exc_info=True)
                raise Exception("Cannot access the shared storage. Please check your storage path.") from e
            if scratch_content != read_content:
                logger.critical(
                    f"Content mismatch detected for path: {scratch_path}.Worker cannot read expected head node content."
                )
                raise Exception("Content mismatch detected. Please check your shared storage setup.")
            num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
            logger.debug(f"=========Starting the training on {rank} with num threads: {num_threads}=========")
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
                random_seed=config["random_seed"],
                **self.kwargs,
            )
            if model_keys:
                model_params = {k: v for k, v in model_params.items() if k in model_keys}
            model = Model(**model_params)
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                devices="auto",
                accelerator=_get_accelerator(),
                strategy=self.ray_trainer_strategy,
                plugins=[ray.train.lightning.RayLightningEnvironment()],
                callbacks=[ray.train.lightning.RayTrainReportCallback()],
                enable_checkpointing=True,
                enable_progress_bar=config.get("enable_progress_bar", True),
            )
            trainer = ray.train.lightning.prepare_trainer(trainer)
            if config.get("worker_mode") == "inference":
                logger.debug("Starting inference mode")
                writer_cb = DistributedPredictionWriter(
                    output_dir=self.result_storage_path, rank=rank, format=config["prediction_format"]
                )
                trainer.callbacks.append(writer_cb)
                trainer.predict(model, datamodule=ann_dm, ckpt_path=ckpt_path)
            else:
                logger.debug("Starting training mode")
                trainer.fit(model, datamodule=ann_dm, ckpt_path=ckpt_path)

        return worker_fn

    def _setup(self,
        file_paths: list[str],
        batch_size: int,
        test_size: float,
        val_size: float,
        prefetch_factor: int,
        max_epochs: int,
        thread_per_worker: int | None,
        num_workers: int | None ,
        result_storage_path: str,
        # read more here: https://lightning.ai/docs/pytorch/stable/common/trainer.html#fit
        ckpt_path: str | None,
        is_gpu: bool,
        random_seed: int | None,
        resource_per_worker: dict | None,
        is_shuffled: bool,
        enable_progress_bar: bool,
        worker_mode: Literal["train", "inference"],
        **kwargs,
        ):
        self.result_storage_path = os.path.expanduser(result_storage_path)
        self.prefetch_factor = prefetch_factor
        self.max_epochs = max_epochs
        self.kwargs = kwargs
        self.enable_progress_bar = enable_progress_bar
        if not resource_per_worker:
            if not thread_per_worker:
                logger.info("Setting thread_per_worker to half of the available CPUs capped at 4")
                thread_per_worker = min(int((self.resources.get("CPU", 2) - 1) / 2), 4)
            resource_per_worker = {"CPU": thread_per_worker}
        if is_gpu and self.resources.get("GPU", 0) == 0:
            logger.warning("`is_gpu = True` but there is no GPU found. Fallback to CPU.")
            is_gpu = False
        if is_gpu:
            if num_workers is None:
                num_workers = int(self.resources.get("GPU"))
            scaling_config = ray.train.ScalingConfig(
                num_workers=num_workers, use_gpu=True, resources_per_worker=resource_per_worker
            )
            resource_per_worker["GPU"] = 1
        else:
            if num_workers is None:
                num_workers = max(int((self.resources.get("CPU", 2) - 1) / thread_per_worker), 1)
            scaling_config = ray.train.ScalingConfig(
                num_workers=num_workers, use_gpu=False, resources_per_worker=resource_per_worker
            )
        logger.info(f"Using {num_workers} workers where each worker uses: {resource_per_worker}")
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
        logger.debug(f"Data splitting time: {time.time() - start:.2f} seconds")
        train_config = {
            "indices": indices,
            "ckpt_path": ckpt_path,
            "shuffle_strategy": shuffle_strategy,
            "enable_progress_bar": self.enable_progress_bar,
            "scratch_path": os.path.join(self.result_storage_path, "scratch.plt"),
            "scratch_content": str(uuid.uuid4()),
            "worker_mode": worker_mode,
            "random_seed": random_seed,
        }
        if worker_mode == "inference":
            train_config["prediction_format"] = kwargs["prediction_format"]
        par_trainer = ray.train.torch.TorchTrainer(
            self._worker_fn(),
            scaling_config=scaling_config,
            train_loop_config=train_config,
            run_config=ray.train.RunConfig(storage_path=self.result_storage_path),
        )

        logger.debug("Writing scratch content to share storage")
        scratch_path = train_config["scratch_path"]
        fs, path_on_fs = fsspec.core.url_to_fs(scratch_path)
        parent_dir = os.path.dirname(path_on_fs)
        if not fs.exists(parent_dir):
            logger.debug(f"Ensuring directory exists: {parent_dir}")
            fs.makedirs(parent_dir, exist_ok=True)
        file = get_fsspec(scratch_path, mode="w")
        file.write(train_config["scratch_content"])
        file.close()
        logger.debug("Spawning Ray worker and initiating distributed training")
        return par_trainer, indices

    @beartype
    def par_inference(
        self,
        file_paths: list[str],
        ckpt_path: str | None = None,
        result_storage_path: str = "~/protoplast_results",
        batch_size: int = 2000,
        prefetch_factor: int = 4,
        thread_per_worker: int | None = None,
        num_workers: int | None = None,
        is_gpu: bool = True,
        resource_per_worker: dict | None = None,
        enable_progress_bar: bool = True,
        prediction_format: Literal["csv", "parquet"] = "csv",
        **kwargs,
    ):
        """Start parallel inference the order of the result is not guaranteed to be the same as input file

        Parameters
        ----------
        file_paths : list[str]
            List of h5ad AnnData files
        batch_size : int, optional
            How much data to fetch from disk, by default to 2000
        prefetch_factor : int, optional
            Total data fetch is prefetch_factor * batch_size, by default 4
        thread_per_worker : int | None, optional
            Amount of worker for each dataloader, by default None
        num_workers : int | None, optional
            Override number of Ray processes default to number of GPU(s) in the cluster, by default None
        is_gpu : bool, optional
            By default True turn this off if your system don't have any GPU, by default True
        resource_per_worker : dict | None, optional
            This get pass to Ray you can specify how much CPU or GPU each Ray process get, by default None
        ckpt_path: str | None = None,
            Path of the checkpoint if this is specified it will train from checkpoint otherwise it will start the
            training from scratch, by default None
        enable_progress_bar : bool
            Whether to enable Trainer progress bar or not, by default True
        Returns
        -------
        Result
            The inference result from RayTrainer
        """
        par_trainer, _ = self._setup(
            file_paths,
            batch_size,
            0.0,
            0.0,
            prefetch_factor,
            1,
            thread_per_worker,
            num_workers,
            result_storage_path,
            ckpt_path,
            is_gpu,
            None,
            resource_per_worker,
            False,
            enable_progress_bar,
            prediction_format=prediction_format,
            worker_mode="inference",
            **kwargs,
        )
        # despite the confusing name we use fit to run inference here
        result = par_trainer.fit()
        # combine the result and order it correctly
        return result
    
    def inference(
            self, 
            file_paths: list[str], 
            result_storage_path: str, 
            ckpt_path: str,
            prediction_format: Literal["csv", "parquet"] = "csv",
            enable_progress_bar: bool = True,
            batch_size=2000
        ):
        """Start inference in a single process order is guarantee to be the same as input file
        don't use this in a distributed cluster
        Parameters
        ----------
        file_paths : list[str]
            List of h5ad AnnData files
        result_storage_path : str
            Path to store the prediction result
        ckpt_path : str
            Path of the checkpoint to run inference
        enable_progress_bar : bool, optional
            Whether to enable Trainer progress bar or not, by default True
        batch_size : int, optional
            How much data to fetch from disk, by default to 2000
        """
        shuffle_strategy = self.shuffle_strategy(
            file_paths,
            batch_size,
            1,
            0,
            0,
            None,
            metadata_cb=self.metadata_cb,
            is_shuffled=False,
            prediction_format=prediction_format,
        )
        indices = shuffle_strategy.split()
        writer_cb = PredictionWriterCallback(output_path=result_storage_path, format=prediction_format)
        trainer = pl.Trainer(
                devices="auto",
                accelerator=_get_accelerator(),
                enable_progress_bar=enable_progress_bar,
            )
        trainer.callbacks.append(writer_cb)
        ann_dm = AnnDataModule(
            indices,
            self.Ds,
            4,
            self.sparse_key,
            SequentialShuffleStrategy,
            self.before_dense_cb,
            self.after_dense_cb,
            batch_size=batch_size,
            is_single_thread=True
        )
        model_params = indices.metadata
        if self.model_keys:
            model_params = {k: v for k, v in model_params.items() if k in self.model_keys}
        model = self.Model(**model_params)
        trainer.predict(model, datamodule=ann_dm, ckpt_path=ckpt_path)
    

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
        enable_progress_bar: bool = True,
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
        enable_progress_bar : bool
            Whether to enable Trainer progress bar or not, by default True
        Returns
        -------
        Result
            The training result from RayTrainer
        """
        par_trainer, _ = self._setup(
            file_paths,
            batch_size,
            test_size,
            val_size,
            prefetch_factor,
            max_epochs,
            thread_per_worker,
            num_workers,
            result_storage_path,
            ckpt_path,
            is_gpu,
            random_seed,
            resource_per_worker,
            is_shuffled,
            enable_progress_bar,
            worker_mode="train",
            **kwargs,)
        return par_trainer.fit()

    

def _get_accelerator() -> Literal["cpu", "auto"]:
    """Get accelerator for RayTrainRunner"""
    if torch.backends.mps.is_available():
        # TODO: Make RayDDPStrategy compatible with MPS
        accelerator = "cpu"
        warnings.warn("RayTrainRunner does not support MPS accelarator yet. Fallback to CPU", UserWarning, stacklevel=2)
    else:
        accelerator = "auto"
    return accelerator
