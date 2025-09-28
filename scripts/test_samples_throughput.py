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


import argparse
import glob
import os
import time

import anndata as ad
import psutil
from torch.utils.data import DataLoader
from tqdm import tqdm

from protoplast.scrna.anndata.strategy import SequentialShuffleStrategy
from protoplast.scrna.anndata.torch_dataloader import DistributedAnnDataset


def get_total_memory_mb() -> float:
    """Return total memory usage of current process and all its children in MB."""
    parent = psutil.Process(os.getpid())
    total_mem = parent.memory_info().rss  # main process

    for child in parent.children(recursive=True):
        try:
            total_mem += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass  # child may have exited

    return total_mem / 1024**2


def benchmark(loader, n_samples, batch_size, max_iteration=None, warmup_iteration=100, sampling_memory_step=5):
    if max_iteration is None:
        # if no max_iteration is provided, we run for the entire dataset
        max_iteration = n_samples // batch_size
    loader_iter = loader.__iter__()

    peak_memory = get_total_memory_mb()
    start_time = time.time()
    batch_times = []
    batch_time = time.time()
    # we warmup for the first warmup_iteration iterations, then we run for max_iteration iterations
    max_iteration += warmup_iteration
    for i, _batch in tqdm(enumerate(loader_iter), total=max_iteration):
        if i < warmup_iteration:
            batch_time = time.time()
            continue
        batch_times.append(time.time() - batch_time)
        batch_time = time.time()
        if i % sampling_memory_step == 0:
            peak_memory = max(peak_memory, get_total_memory_mb())
        if i == max_iteration:
            break

    execution_time = time.time() - start_time
    time_per_sample = (1e6 * execution_time) / (max_iteration * batch_size)
    samples_per_sec = max_iteration * batch_size / execution_time

    return samples_per_sec, time_per_sample, batch_times, peak_memory


def pass_through_collate_fn(batch):
    return batch[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_glob", type=str, help="glob pattern of the h5 files")
    parser.add_argument("--max_iter", dest="max_iteration", default=5000, type=int)
    parser.add_argument("--batch_size", dest="batch_size", default=64, type=int)
    parser.add_argument("--n_workers", dest="n_workers", default=32, type=int)
    parser.add_argument("--pre_fetch_then_batch", dest="pre_fetch_then_batch", default=8, type=int)
    parser.add_argument(
        "--warmup_iter",
        dest="warmup_iteration",
        default=100,
        type=int,
        help="# of first iterations will be ignored when benchmarking",
    )  # noqa: E501
    parser.add_argument(
        "--sampling_memory_step",
        dest="sampling_memory_step",
        default=1000,
        type=int,
        help="# of iterations between sampling memory",
    )  # noqa: E501
    args = parser.parse_args()

    print("=== PARAMETERS ===")
    print(f"data_glob={args.data_glob}")
    print(f"batch_size={args.batch_size}")
    print(f"n_workers={args.n_workers}")
    print(f"warmup_iterations={args.warmup_iteration}")
    print(f"max_iterations={args.max_iteration}")
    print(f"pre_fetch_then_batch={args.pre_fetch_then_batch}")

    print("=== PROGRESS ===")
    N_WORKERS = args.n_workers
    PREFETCH_FACTOR = 16
    # Example how to test throughput with DistributedAnnDataset
    files = glob.glob(args.data_glob)

    n_cells = 0

    for file in files:
        n_cells += ad.read_h5ad(file, backed="r").n_obs

    shuffle_strategy = SequentialShuffleStrategy(
        files,
        batch_size=args.batch_size,
        total_workers=args.n_workers,
        test_size=0.0,
        validation_size=0.0,
        pre_fetch_then_batch=args.pre_fetch_then_batch,
    )

    indices = shuffle_strategy.split()
    ds = DistributedAnnDataset.create_distributed_ds(indices, sparse_key="X")
    dataloader = DataLoader(
        ds,
        batch_size=None,
        num_workers=N_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=False,
        persistent_workers=False,
    )
    samples_per_sec, time_per_sample, batch_times, peak_memory = benchmark(
        dataloader,
        n_cells,
        args.batch_size,
        max_iteration=args.max_iteration,
        warmup_iteration=args.warmup_iteration,
        sampling_memory_step=args.sampling_memory_step,
    )

    print("=== RESULT ===")
    print(f"samples per sec: {samples_per_sec:.2f} samples/sec")
    print(f"time per sample: {time_per_sample:.2f} μs")
    print(f"peak memory: {peak_memory:.2f} MB")


if __name__ == "__main__":
    main()
