from typing import Callable
import anndata
from torch.utils.data._utils.collate import default_collate
import torch
import random
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

def ann_split_data(
    file_paths: list[str],
    batch_size: int,
    total_workers: int,
    test_size: float | None = None,
    validation_size: float | None = None,
    rng: random.Random | None = None,
    metadata_cb: Callable[[anndata.AnnData, dict], None] | None = None,
    is_shuffled: bool = True,
):
    def to_batches(n):
        return [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    if not rng:
        rng = random.Random()

    file_batches = []
    total_batches = 0
    metadata = dict()

    # First pass: compute per-file batches
    for i, fp in enumerate(file_paths):
        ad = anndata.read_h5ad(fp, backed="r")
        if i == 0 and metadata_cb:
            metadata_cb(ad, metadata)

        n_obs = ad.n_obs
        if batch_size > n_obs:
            warnings.warn(
                f"Batch size ({batch_size}) is greater than number of observations "
                f"in file {fp} ({n_obs}). Only one batch will be created.",
                stacklevel=2,
            )

        batches = to_batches(n_obs)

        # Drop per-file remainder to make divisible by total_workers
        remainder = len(batches) % total_workers
        if remainder > 0:
            batches = batches[:-remainder]

        total_batches += len(batches)
        file_batches.append(batches)

    # Safety check
    if (test_size or 0) + (validation_size or 0) > 1:
        raise ValueError("test_size + validation_size must be <= 1")

    # How many batches should go to validation & test globally?
    val_total = int(total_batches * validation_size) if validation_size else 0
    test_total = int(total_batches * test_size) if test_size else 0

    train_datas, validation_datas, test_datas = [], [], []

    # Second pass: allocate splits proportionally per file
    for batches in file_batches:
        if is_shuffled:
            rng.shuffle(batches)
        n = len(batches)

        val_n = int(round(n / total_batches * val_total)) if validation_size else 0
        test_n = int(round(n / total_batches * test_total)) if test_size else 0

        val_split = batches[:val_n]
        test_split = batches[val_n : val_n + test_n]
        train_split = batches[val_n + test_n :]

        validation_datas.append(val_split)
        test_datas.append(test_split)
        train_datas.append(train_split)

    return dict(
        files=file_paths,
        train_indices=train_datas,
        val_indices=validation_datas,
        test_indices=test_datas,
        metadata=metadata,
    )

@dataclass
class SplitInfo:
    files: list[str]
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    metadata: dict[str, any]

    def to_dict(self) -> dict[str, any]:
        return {
            "files": self.files,
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "metadata": self.metadata
        }


class ShuffleStrategy(ABC):
    def __init__(self, 
            file_paths: list[str],
            batch_size: int,
            mini_batch_size: int,
            total_workers: int,
            test_size: float | None = None,
            validation_size: float | None = None,
            random_seed: int | None = 42,
            metadata_cb: Callable[[anndata.AnnData, dict], None] | None = None,
            is_shuffled: bool = True,
        ) -> None:
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.total_workers = total_workers
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_seed = random_seed
        self.metadata_cb = metadata_cb
        self.is_shuffled = is_shuffled
        self.rng = random.Random(random_seed) if random_seed else random.Random()

    @abstractmethod
    def split(self) -> SplitInfo:
        pass
    @abstractmethod
    def mixer(self, batch: list) -> any:
        pass
    

class DefaultShuffleStrategy(ShuffleStrategy):
    def __init__(self, file_paths: list[str],
            batch_size: int,
            mini_batch_size: int,
            total_workers: int,
            test_size: float | None = None,
            validation_size: float | None = None,
            random_seed: int | None = 42,
            metadata_cb: Callable[[anndata.AnnData, dict], None] | None = None,
            is_shuffled: bool = True,) -> None:
        super().__init__(file_paths, batch_size, mini_batch_size, total_workers, test_size, validation_size, random_seed, metadata_cb, is_shuffled)

    def split(self) -> SplitInfo:
        split_dict = ann_split_data(
            self.file_paths,
            self.batch_size,
            self.total_workers,
            self.test_size,
            self.validation_size,
            self.rng,
            self.metadata_cb,
            self.is_shuffled
        )
        return SplitInfo(**split_dict)
    

    def mixer(self, batch: list):
        self.rng.shuffle(batch)

        def collate_item(items):
            sample = items[0]

            # Sparse tensor
            if isinstance(sample, torch.Tensor) and (sample.is_sparse or sample.is_sparse_csr):
                return torch.vstack(items)

            # Dense tensor
            elif isinstance(sample, torch.Tensor):
                return default_collate(items)

            # Dict
            elif isinstance(sample, dict):
                return {k: collate_item([d[k] for d in items]) for k in sample}

            # Tuple or list
            elif isinstance(sample, (tuple, list)):
                return type(sample)(collate_item([b[i] for b in items]) for i in range(len(sample)))

            # Fallback
            else:
                return default_collate(items)

        return collate_item(batch)