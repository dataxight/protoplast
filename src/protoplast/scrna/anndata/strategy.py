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


import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import anndata
import torch
from torch.utils.data._utils.collate import default_collate


def ann_split_data(
    file_paths: list[str],
    batch_size: int,
    total_workers: int,
    test_size: float | None = None,
    validation_size: float | None = None,
    rng: random.Random | None = None,
    metadata_cb: Callable[[anndata.AnnData, dict], None] | None = None,
    is_shuffled: bool = True,
    drop_last: bool = True,
    drop_remainder_warning_pct: float = 0.05,
):
    def to_batches(n):
        batches = []
        for i in range(0, n, batch_size):
            if i + batch_size > n:
                continue
            batches.append((i, i + batch_size))
        return batches

    def drop_remainder(batches):
        n = sum(end - start for start, end in batches)
        remainder = len(batches) % total_workers
        if remainder > 0:
            if drop_last:
                dropped_n = sum(end - start for start, end in batches[-remainder:])
                if dropped_n / n > drop_remainder_warning_pct:
                    print(f"=========Warning: {dropped_n / n} of data is dropped")
                batches = batches[:-remainder]
            else:
                pad = total_workers - remainder
                print("Duplicating data with", pad)
                batches.extend(batches[-remainder:] * (pad // remainder))
                batches.extend(batches[-(pad % remainder) :])
        return batches

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

        # very extreme case, that we have number of batches less than total_workers.
        # then we have to pad using the last range to make it divisible by total_workers
        if len(batches) < total_workers:
            batches += [batches[-1]] * (total_workers - len(batches))
        if len(batches) == 0:
            raise Exception("This data is not compatiable with this worker combination")

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
        print(
            "=========Length of val_split",
            len(val_split),
            "length of test_split",
            len(test_split),
            "length of train_split",
            len(train_split),
        )
        val_split = drop_remainder(val_split)
        test_split = drop_remainder(test_split)
        train_split = drop_remainder(train_split)
        print(
            "=========Length of after dropping remainder val_split",
            len(val_split),
            "length of test_split",
            len(test_split),
            "length of train_split",
            len(train_split),
        )

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
    train_indices: list[list[int]]
    val_indices: list[list[int]]
    test_indices: list[list[int]]
    metadata: dict[str, any]
    mini_batch_size: int | None = None
    """Information on how to split the data
    this will get pass to the Dataset to know which part of the data
    they need to access

    Parameters
    ----------
    files : list[str]
        List of files
    train_indices : list[list[str]]
        List of indices for training `train_indices[file_idx][batch_idx]` where `file_idx` must correspond
        to the idx of `files` parameter
    val_indices : list[list[str]]
        List of indices for validation `val_indices[file_idx][batch_idx]` where `file_idx` must correspond
        to the idx of `files` parameter
    test_indices : list[list[str]]
        List of indices for testing `test_indices[file_idx][batch_idx]` where `file_idx` must correspond
        to the idx of `files` parameter
    metadata : dict[str, any]
        Data to pass on to the Dataset and model
    mini_batch_size : int | None
        How much data to send to the model
    """

    def to_dict(self) -> dict[str, any]:
        return {
            "files": self.files,
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "metadata": self.metadata,
            "mini_batch_size": self.mini_batch_size,
        }


class ShuffleStrategy(ABC):
    """Strategy on how to data should be split and shuffle during
    the training

    Parameters
    ----------
    file_paths : list[str]
        List of file paths
    batch_size : int
        How much data to fetch
    total_workers : int
        Total workers this is equal to number of processes times number of threads per process
    test_size : float | None, optional
        Fraction of test data for example 0.1 means 10% will be split for testing, by default None
    validation_size : float | None, optional
        Fraction of validation data for example 0.2 means 20% will be split for validation, by default None
    random_seed : int | None, optional
        Seed to randomize the split set this to None if you want this to be completely random, by default 42
    metadata_cb : Callable[[anndata.AnnData, dict], None] | None, optional
        Callback to mutate metadata recommended for passing data from `obs` or `var`
        or any additional data your models required
        by default cell_line_metadata_cb
    is_shuffled : bool, optional
        Whether to shuffle the data or not this will be deprecated soon, by default True
    """

    def __init__(
        self,
        file_paths: list[str],
        batch_size: int,
        total_workers: int,
        test_size: float | None = None,
        validation_size: float | None = None,
        random_seed: int | None = 42,
        metadata_cb: Callable[[anndata.AnnData, dict], None] | None = None,
        is_shuffled: bool = True,
    ) -> None:
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.total_workers = total_workers
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_seed = random_seed
        self.metadata_cb = metadata_cb
        self.is_shuffled = is_shuffled
        self.rng = random.Random(random_seed) if random_seed else random.Random()

    @property
    def is_mixer(self):
        return False

    @abstractmethod
    def split(self) -> SplitInfo:
        """
        How you want to split the data in each worker must return SplitInfo
        """
        pass

    @abstractmethod
    def mixer(self, batch: list) -> any:
        """
        If your Dataset only return 1 sample and not prebatched
        this need to be implemented
        """
        pass


class SequentialShuffleStrategy(ShuffleStrategy):
    """Return the data in a sequential way randomness is not guarantee
    there is a high chance the data will come from nearby rows this might
    affect your training accuracy depending on how the anndata are ordered you can
    overcome this by preshuffling the data manually yourself if this is an issue

    Parameters
    ----------
    file_paths : list[str]
        List of file paths
    batch_size : int
        How much data to fetch
    total_workers : int
        Total workers this is equal to number of processes times number of threads per process
    test_size : float | None, optional
        Fraction of test data for example 0.1 means 10% will be split for testing, by default None
    validation_size : float | None, optional
        Fraction of validation data for example 0.2 means 20% will be split for validation, by default None
    random_seed : int | None, optional
        Seed to randomize the split set this to None if you want this to be completely random, by default 42
    metadata_cb : Callable[[anndata.AnnData, dict], None] | None, optional
        Callback to mutate metadata recommended for passing data from `obs` or `var`
        or any additional data your models required
        by default cell_line_metadata_cb
    is_shuffled : bool, optional
        Whether to shuffle the data or not this will be deprecated soon, by default True
    pre_fetch_then_batch : int | None
        The prefetch factor the total size of data fetch will be equal to `pre_fetch_then_batch * batch_size`
    drop_last : bool
        If there is true drop the remainder, default to True otherwise duplicate the data to make sure the
        data is evenly distributed to all the workers
    """

    def __init__(
        self,
        file_paths: list[str],
        batch_size: int,
        total_workers: int,
        test_size: float | None = None,
        validation_size: float | None = None,
        random_seed: int | None = 42,
        metadata_cb: Callable[[anndata.AnnData, dict], None] | None = None,
        is_shuffled: bool = False,
        pre_fetch_then_batch: int | None = 16,
        drop_last: bool = True,
    ) -> None:
        super().__init__(
            file_paths,
            batch_size,
            total_workers,
            test_size,
            validation_size,
            random_seed,
            metadata_cb,
            is_shuffled,
        )
        self.pre_fetch_then_batch = pre_fetch_then_batch
        self.drop_last = drop_last

    def split(self) -> SplitInfo:
        if self.pre_fetch_then_batch:
            batch_size = self.batch_size * self.pre_fetch_then_batch
        else:
            batch_size = self.batch_size
        split_dict = ann_split_data(
            self.file_paths,
            batch_size,
            self.total_workers,
            self.test_size,
            self.validation_size,
            self.rng,
            self.metadata_cb,
            self.is_shuffled,
            self.drop_last,
        )
        # this will be passed to the dataset, inorder to know the mini batch size
        if self.pre_fetch_then_batch:
            split_dict["mini_batch_size"] = self.batch_size
        else:
            # yield everything we read
            split_dict["mini_batch_size"] = None
        return SplitInfo(**split_dict)

    def mixer(self, batch: list):
        return super().mixer(batch)


class RandomShuffleStrategy(ShuffleStrategy):
    def __init__(
        self,
        file_paths: list[str],
        batch_size: int,
        mini_batch_size: int,
        total_workers: int,
        test_size: float | None = None,
        validation_size: float | None = None,
        random_seed: int | None = 42,
        metadata_cb: Callable[[anndata.AnnData, dict], None] | None = None,
        is_shuffled: bool = True,
        drop_last: bool = True,
    ) -> None:
        super().__init__(
            file_paths,
            batch_size,
            total_workers,
            test_size,
            validation_size,
            random_seed,
            metadata_cb,
            is_shuffled,
        )
        self.mini_batch_size = mini_batch_size
        self.drop_last = drop_last

    def split(self) -> SplitInfo:
        split_dict = ann_split_data(
            self.file_paths,
            self.batch_size,
            self.total_workers,
            self.test_size,
            self.validation_size,
            self.rng,
            self.metadata_cb,
            self.is_shuffled,
            self.drop_last,
        )
        return SplitInfo(**split_dict)

    @property
    def is_mixer(self):
        return True

    def mixer(self, batch: list):
        self.rng.shuffle(batch)

        def collate_item(items):
            sample = items[0]

            # Sparse tensor
            if isinstance(sample, torch.Tensor) and (sample.is_sparse or sample.is_sparse_csr):
                return torch.stack(items)

            # Dense tensor
            elif isinstance(sample, torch.Tensor):
                return default_collate(items)

            # Dict
            elif isinstance(sample, dict):
                return {k: collate_item([d[k] for d in items]) for k in sample}

            # Tuple or list
            elif isinstance(sample, (tuple | list)):
                return type(sample)(collate_item([b[i] for b in items]) for i in range(len(sample)))

            # Fallback
            else:
                return default_collate(items)

        return collate_item(batch)
