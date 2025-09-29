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


# This script performs benchmarking for PROTOplast

# WARNING: DO NOT IMPORT PROTOPLAST EARLY ON IF YOU ARE GOING TO
#          BENCHMARK NON-PROTOPLAST LIBRARIES. IMPORTING PROTOPLAST
#          WILL APPLY PATCHING TO ANNDATA AND CAUSE AN ERROR WHEN
#          MULTIPROCESSING.

# === LIBS ===

# General
import argparse
import csv
import glob
import os
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from typing import Literal, get_args

# Data structure
import anndata as ad
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import psutil
import pynvml
import scanpy as sc

# AI/ML
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata.experimental import AnnCollection, AnnLoader
from pydantic import BaseModel

# scDataset
from scdataset import Streaming, scDataset
from scipy import sparse

# SCVI
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.dataloaders import AnnDataLoader
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# === HELPER FUNCTIONS ===


class Classifier(pl.LightningModule):
    """
    A simple classification model

    This model will be use for all benchmarking runners.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if x.is_sparse or x.is_sparse_csr:
            x = x.to_dense()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


# === TYPES ===

RunnerClass = Literal["anndata", "protoplast", "scvi", "scvi2", "scdataset", "annloader"]


class BenchmarkRunnerParams(BaseModel):
    glob: str
    name: str
    label: str
    batch_size: int
    fetch_factor: int
    num_workers: int  # Per GPU
    class_name: RunnerClass
    logfile: str
    num_gpus: int


# === BENCHMARKING FUNCTIONS ===


class BenchmarkRunner(ABC):
    def __init__(self, params: BenchmarkRunnerParams):
        self.params = params
        self.adata_paths = glob.glob(params.glob)

        if len(self.adata_paths) == 0:
            raise FileNotFoundError(f"Invalid glob: {self.params.glob}")

        # Get data name
        self.data_name = os.path.basename(params.glob)
        if self.data_name.startswith("*"):
            self.data_name = os.path.join(os.path.basename(os.path.dirname(params.glob)), self.data_name)

        adatas = [ad.read_h5ad(p, backed="r") for p in self.adata_paths]
        self.cell_count = sum(x.obs.shape[0] for x in adatas)

        self.logfile = Path(params.logfile)
        # create header if file does not exist
        if not self.logfile.exists():
            with self.logfile.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Timestamp",
                        "Class",
                        "Data",
                        "Cell_count",
                        "Batch size",
                        "Fetch factor",
                        "# workers per GPU",
                        "# GPUs",
                        "Task",
                        "Elapsed (seconds)",
                        "Peak RAM usage (MB)",
                        "Peak GPU usage (MB)",
                    ]
                )

    @contextmanager
    def record(self, msg: str):
        success = True

        # Start recording RAM usage
        stop_event = Event()
        peak = {"rss": psutil.virtual_memory().used}

        # Start recording GPU usage
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        peak["gpu"] = 0

        def monitor():
            """Monitor both GPU and RAM usage"""
            while not stop_event.is_set():
                # RAM
                rss = psutil.virtual_memory().used
                if rss > peak["rss"]:
                    peak["rss"] = rss

                # GPU
                gpu_used = 0
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_used += meminfo.used
                if gpu_used > peak["gpu"]:
                    peak["gpu"] = gpu_used

                time.sleep(0.5)

        thread = Thread(target=monitor, daemon=True)
        thread.start()

        # Start counting time
        start = time.perf_counter()

        try:
            yield
        except Exception:
            success = False
            traceback.print_exc()
        finally:
            # Stop counting time
            end = time.perf_counter()
            elapsed = end - start

            # Stop monitoring RAM and GPU
            stop_event.set()
            thread.join()
            pynvml.nvmlShutdown()

            with self.logfile.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        round(time.time()),
                        self.params.class_name,
                        self.data_name,
                        self.cell_count,
                        self.params.batch_size,
                        self.params.fetch_factor,
                        self.params.num_workers,
                        self.params.num_gpus,
                        msg,
                        elapsed if success else -1,
                        peak["rss"] / 1024**2,
                        peak["gpu"] / 1024**2,
                    ]
                )
            if success:
                print(f"{msg} took {elapsed:.2f}s")
            else:
                print(f"{msg} failed after {elapsed:.2f}s")
            print(f"{msg} peak RAM: {peak['rss'] / 1024**2:.2f} MB")
            print(f"{msg} peak GPU: {peak['gpu'] / 1024**2:.2f} MB")

    def run(self):
        with self.record("End-to-end"):
            self._run()

    @abstractmethod
    def _run(self):
        pass


class ScDatasetRunner(BenchmarkRunner):
    def _run(self):
        """Ref: https://github.com/scDataset/scDataset/blob/main/tahoe_tutorial.ipynb"""

        def fetch_transform(batch) -> ad.AnnData:
            # Materialize the AnnData batch (X matrix) in memory
            return batch.to_adata()

        def batch_transform(batch: ad.AnnData):
            # Convert AnnData batch to tensors for training
            X = batch.X.astype("float32")
            # Densify if X is a sparse matrix
            if sparse.issparse(X):
                X = X.toarray()
            X = torch.from_numpy(X)
            y = torch.tensor(batch.obs[self.params.label].astype("category").cat.codes.to_numpy(), dtype=torch.long)
            return X, y

        collection = AnnCollection([ad.read_h5ad(p, backed="r") for p in self.adata_paths])
        strategy = Streaming(indices=np.arange(collection.n_obs), shuffle=False)
        ds = scDataset(
            data_collection=collection,
            strategy=strategy,
            batch_size=self.params.batch_size,
            fetch_factor=self.params.fetch_factor,
            fetch_transform=fetch_transform,
            batch_transform=batch_transform,
        )
        dataloader = DataLoader(
            ds,
            batch_size=None,
            num_workers=self.params.num_workers,
            pin_memory=True,
            prefetch_factor=self.params.fetch_factor + 1,
        )
        model = Classifier(
            input_dim=collection.n_vars,
            hidden_dim=128,
            num_classes=collection.obs[self.params.label].nunique(),
        )
        trainer = pl.Trainer(max_epochs=1, accelerator="auto", devices=self.params.num_gpus)
        trainer.fit(model, dataloader)


class AnndataRunner(BenchmarkRunner):
    """Benchmark Standard AnnData"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(self.adata_paths) > 1:
            raise RuntimeError("This class cannot handle multiple H5 files")

    def _run(self):
        class StandardAnnDataset(Dataset):
            """Use standard AnnData object"""

            def __init__(self, adata: ad.AnnData, label: str):
                X = adata.X
                if sparse.issparse(X):
                    self.X = X.toarray()
                else:
                    self.X = np.asarray(X)
                self.y = pd.Categorical(adata.obs[label]).codes

            def __len__(self) -> int:
                return self.X.shape[0]

            def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.tensor(self.X[idx], dtype=torch.float32)
                y = torch.tensor(self.y[idx], dtype=torch.long)
                return x, y

        adata = sc.read_h5ad(self.adata_paths[0])
        dataset = StandardAnnDataset(adata, self.params.label)

        # Init loader, model, and trainer
        dataloader = DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            prefetch_factor=self.params.fetch_factor + 1,
        )
        model = Classifier(
            input_dim=adata.var.shape[0],
            hidden_dim=128,
            num_classes=adata.obs[self.params.label].nunique(),
        )
        # Train
        trainer = pl.Trainer(max_epochs=1, accelerator="auto", devices=self.params.num_gpus)
        trainer.fit(model, dataloader)


class AnnLoaderRunner(BenchmarkRunner):
    def _run(self):
        def collate_fn(batch) -> torch.Tensor:
            X = torch.stack([b.X.view(-1) for b in batch])
            y = torch.stack(
                [
                    torch.tensor(
                        b.obs[self.params.label].astype("category").cat.codes.to_numpy(), dtype=torch.long
                    ).view(-1)
                    for b in batch
                ],
                axis=1,
            )
            return X, y[0]

        collection = AnnCollection([ad.read_h5ad(p, backed="r") for p in self.adata_paths])
        dataloader = AnnLoader(
            collection,
            batch_size=self.params.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.params.num_workers,
            prefetch_factor=self.params.fetch_factor + 1,
        )
        # return dataloader
        model = Classifier(
            input_dim=collection.n_vars,
            hidden_dim=128,
            num_classes=collection.obs[self.params.label].nunique(),
        )
        trainer = pl.Trainer(max_epochs=1, accelerator="auto", devices=self.params.num_gpus)
        trainer.fit(model, dataloader)


class SCVIAnnLoaderRunner(BenchmarkRunner):
    def __init__(self, *args, **kwargs):
        """This runner uses SCVI with `load_sparse_tensor=False`"""
        super().__init__(*args, **kwargs)

        if len(self.adata_paths) > 1:
            raise RuntimeError("This class cannot handle multiple H5 files")

    def _run(self):
        def collate_fn(batch):
            y = torch.tensor(batch["label"], dtype=torch.long).view(-1)
            X = torch.from_numpy(batch["counts"]).float()
            return X, y

        adata_manager = AnnDataManager(
            fields=[
                LayerField("counts", layer=None),
                CategoricalObsField("label", self.params.label),
            ]
        )
        adata = ad.read_h5ad(self.adata_paths[0], backed="r")
        adata_manager.register_fields(adata)
        dataloader = AnnDataLoader(
            adata_manager=adata_manager,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            prefetch_factor=self.params.fetch_factor + 1,
            pin_memory=False,
            persistent_workers=False,
            load_sparse_tensor=False,  # dense matrix are used for all data loader
            collate_fn=collate_fn,
        )

        model = Classifier(
            input_dim=adata.n_vars,
            hidden_dim=128,
            num_classes=adata.obs[self.params.label].nunique(),
        )
        trainer = pl.Trainer(max_epochs=1, accelerator="auto", devices=self.params.num_gpus)
        trainer.fit(model, dataloader)


class SCVIAnnLoader2Runner(BenchmarkRunner):
    """This runner uses SCVI with `load_sparse_tensor=True`"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(self.adata_paths) > 1:
            raise RuntimeError("This class cannot handle multiple H5 files")

    def _run(self):
        def collate_fn(batch):
            y = torch.tensor(batch["label"], dtype=torch.long).view(-1)
            return batch["counts"], y

        adata_manager = AnnDataManager(
            fields=[
                LayerField("counts", layer=None),
                CategoricalObsField("label", self.params.label),
            ]
        )
        adata = ad.read_h5ad(self.adata_paths[0], backed="r")
        adata_manager.register_fields(adata)
        dataloader = AnnDataLoader(
            adata_manager=adata_manager,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            prefetch_factor=self.params.fetch_factor + 1,
            pin_memory=False,
            persistent_workers=False,
            load_sparse_tensor=True,
            collate_fn=collate_fn,
        )

        model = Classifier(
            input_dim=adata.n_vars,
            hidden_dim=128,
            num_classes=adata.obs[self.params.label].nunique(),
        )
        trainer = pl.Trainer(max_epochs=1, accelerator="auto", devices=self.params.num_gpus)
        trainer.fit(model, dataloader)


class ProtoplastRunner(BenchmarkRunner):
    """Benchmark PROTOplast"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from protoplast.scrna.anndata.torch_dataloader import DistributedAnnDataset
        from protoplast.scrna.anndata.trainer import RayTrainRunner

        self.DistributedAnnDataset = DistributedAnnDataset
        self.RayTrainRunner = RayTrainRunner

    def _run(self):
        label = self.params.label

        class MyDataset(self.DistributedAnnDataset):
            def transform(self, start: int, end: int):
                X = super().transform(start, end)
                y = torch.tensor(self.ad.obs[label].iloc[start:end].cat.codes.to_numpy(), dtype=torch.long)
                return X, y

        def my_callback(ad: ad.AnnData, metadata: dict):
            metadata["input_dim"] = ad.var.shape[0]
            metadata["hidden_dim"] = 128
            metadata["num_classes"] = ad.obs[label].nunique()

        trainer = self.RayTrainRunner(
            Classifier,
            MyDataset,
            ["input_dim", "hidden_dim", "num_classes"],
            my_callback,
        )
        trainer.train(
            file_paths=self.adata_paths,
            batch_size=self.params.batch_size,
            test_size=0.0,
            val_size=0.0,
            thread_per_worker=self.params.num_workers
            - 1,  # Ray will +1 (https://dataxight.atlassian.net/browse/PROTO-22)
            num_workers=self.params.num_gpus,
            is_shuffled=False,
            max_epochs=1,
            pre_fetch_then_batch=self.params.fetch_factor,
            prefetch_factor=self.params.fetch_factor + 1,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="glob", help="Data glob", type=str)
    parser.add_argument("--class", dest="class_name", default="protoplast", choices=get_args(RunnerClass))
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1024)
    parser.add_argument("--fetch-factor", dest="fetch_factor", type=int, default=16)
    parser.add_argument("--workers", dest="num_workers", type=int, default=12)
    parser.add_argument("--label", dest="label", type=str, default="cell_line")
    parser.add_argument("--logfile", dest="logfile", type=str, default="benchmark_log.csv")
    parser.add_argument("--gpus", dest="num_gpus", type=int, default=1)
    params: BenchmarkRunnerParams = parser.parse_args()

    print("=== PARAMS ===")
    print(params)

    match params.class_name:
        case "anndata":
            Runner = AnndataRunner
        case "protoplast":
            Runner = ProtoplastRunner
        case "scvi":
            Runner = SCVIAnnLoaderRunner
        case "scvi2":
            Runner = SCVIAnnLoader2Runner
        case "scdataset":
            Runner = ScDatasetRunner
        case "annloader":
            Runner = AnnLoaderRunner
        case _:
            parser.print_help()
            exit(1)

    print("=== PROGRESS ===")
    runner = Runner(params)
    runner.run()
