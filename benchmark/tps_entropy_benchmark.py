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
import gc
import os
import time

import numpy as np
import pandas as pd
import psutil
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from protoplast.scrna.anndata.strategy import RandomShuffleStrategy, SequentialShuffleStrategy
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


def evaluate_loader(loader, test_time_seconds=120, description="Testing loader"):
    """Evaluate the performance of a data loader for a specified duration."""
    gc.collect()

    total_samples = 0
    batch_plates = []

    pbar = tqdm(desc=f"{description} (for {test_time_seconds}s)")

    # Initialize warm-up timer
    warm_up_seconds = 30
    warm_up_start = time.perf_counter()
    warm_up_end = warm_up_start + warm_up_seconds
    is_warming_up = True
    total_memories = []

    for batch in loader:
        # Handle different batch structures
        X, plate = batch
        batch_size = X.shape[0]
        if not is_warming_up:
            # Collect plate info for entropy calculation
            batch_plates.append(plate)

        current_time = time.perf_counter()

        if is_warming_up:
            # We're in warm-up period
            if current_time >= warm_up_end:
                # Warm-up complete, start the actual timing
                is_warming_up = False
                total_samples = 0
                start_time = time.perf_counter()
                end_time = start_time + test_time_seconds
                pbar.set_description(f"{description} (warming up complete, testing for {test_time_seconds}s)")
            else:
                pbar.set_description(
                    f"{description}(warming up: {{current_time - warm_up_start:.1f}}/{{warm_up_seconds}}s)"
                )
                pbar.update(1)
                continue

        # Now we're past the warm-up period
        total_samples += batch_size

        elapsed = current_time - start_time
        pbar.set_postfix(samples=total_samples, elapsed=f"{elapsed:.2f}s")
        pbar.update(1)
        # only collect memory after warm up
        total_memories.append(get_total_memory_mb())

        if current_time >= end_time:
            break

    pbar.close()

    # Calculate the load time metrics
    elapsed = time.perf_counter() - start_time
    avg_time_per_sample = elapsed / total_samples if total_samples > 0 else 0
    samples_per_second = total_samples / elapsed if elapsed > 0 else 0

    # Calculate entropy measures (if plate data is available)
    avg_batch_entropy = 0
    std_batch_entropy = 0
    if batch_plates:
        batch_entropies = []
        # Calculate entropy for each batch
        for plates in batch_plates:
            if len(plates) > 1:
                _, counts = np.unique(plates, return_counts=True)
                probabilities = counts / len(plates)
                batch_entropy = stats.entropy(probabilities, base=2)
                batch_entropies.append(batch_entropy)

        # Calculate average and standard deviation of entropy across all batches
        if batch_entropies:
            avg_batch_entropy = np.mean(batch_entropies)
            std_batch_entropy = np.std(batch_entropies)
    max_memory_usage = max(*total_memories)
    average_memory_usage = sum(total_memories) / len(total_memories)

    return {
        "samples_tested": total_samples,
        "elapsed": elapsed,
        "avg_time_per_sample": avg_time_per_sample,
        "samples_per_second": samples_per_second,
        "avg_batch_entropy": avg_batch_entropy,
        "std_batch_entropy": std_batch_entropy,
        "max_memory_usage": max_memory_usage,
        "average_memory_usage": average_memory_usage,
    }


def save_results_to_csv(results, filepath=None):
    """Save or update results to CSV file."""

    df = pd.DataFrame(results)

    # Save to CSV
    if filepath is not None:
        df.to_csv(filepath, index=False)
        print(f"Updated results saved to {filepath}")

    return df


def run_random(batch_size, mini_batch_size, num_workers, prefetch_factor, paths, test_time, max_open_file):
    # Initialize shuffle strategy and split data
    shuffle_strategy = RandomShuffleStrategy(
        paths,
        batch_size,
        mini_batch_size,
        total_workers=num_workers,
        test_size=0.0,
        validation_size=0.0,
        is_shuffled=True,
    )

    indices = shuffle_strategy.split()

    class BenchmarkDistributedAnnDataset(DistributedAnnDataset):
        def transform(self, start: int, end: int):
            X = super().transform(start, end)
            plate = self.ad.obs["plate"].iloc[start:end]
            if X is None:
                return None
            return X, plate

    # Initialize dataset and dataloader
    dataset = BenchmarkDistributedAnnDataset(
        file_paths=paths,
        indices=indices.train_indices,
        metadata=indices.metadata,
        sparse_keys=["X"],
        max_open_files=max_open_file,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=shuffle_strategy.mini_batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor + 1,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=shuffle_strategy.mixer,
    )

    # Evaluate the dataloader performance
    results = evaluate_loader(dataloader, test_time_seconds=test_time, description="DataLoader Benchmark")
    return {
        "batch_size": batch_size,
        "mini_batch_size": mini_batch_size,
        "num_workers": num_workers,
        "max_open_files": max_open_file,
        **results,
    }


def run_sequential(batch_size, num_workers, prefetch_factor, paths, test_time):
    # Initialize shuffle strategy and split data
    shuffle_strategy = SequentialShuffleStrategy(
        paths,
        batch_size,
        total_workers=1,  # the total worker here is ray worker only
        test_size=0.0,
        validation_size=0.0,
        is_shuffled=True,
    )

    indices = shuffle_strategy.split()

    class BenchmarkDistributedAnnDataset(DistributedAnnDataset):
        def transform(self, start: int, end: int):
            X = super().transform(start, end)
            plate = self.ad.obs["plate"].iloc[start:end]
            if X is None:
                return None
            return X, plate

    # Initialize dataset and dataloader
    dataset = BenchmarkDistributedAnnDataset(
        file_paths=paths,
        indices=indices.train_indices,
        metadata=indices.metadata,
        sparse_keys=["X"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor + 1,
        persistent_workers=True,
        pin_memory=True,
    )

    # Evaluate the dataloader performance
    results = evaluate_loader(dataloader, test_time_seconds=test_time, description="DataLoader Benchmark")
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        **results,
    }


def test_random(paths, args):
    batch_sizes = [500, 1500, 2000]
    mini_batch_sizes = [50, 100, 250]
    num_workers = [2, 4]
    max_open_files = [6, 9, 12]
    prefetch_factor = 2
    results = []
    for batch_size in batch_sizes:
        for mini_batch_size in mini_batch_sizes:
            for num_worker in num_workers:
                for max_open_file in max_open_files:
                    print(
                        f"Running random benchmark with batch_size={batch_size},"
                        "mini_batch_size={mini_batch_size}, num_workers={num_worker}, "
                        "prefetch_factor={prefetch_factor}, max_open_file={max_open_file}"
                    )
                    results.append(
                        run_random(
                            batch_size,
                            mini_batch_size,
                            num_worker,
                            prefetch_factor,
                            paths,
                            args.test_time,
                            max_open_file,
                        )
                    )
                    save_results_to_csv(results, args.output_csv)


def test_sequential(paths, args):
    # batch_sizes = [250, 500, 1500, 2000]
    # num_workers = [2, 4, 8, 16]
    batch_sizes = [2500, 3000, 4000, 5000, 10000]
    num_workers = [8, 10, 12]
    prefetch_factor = 4
    results = []
    for batch_size in batch_sizes:
        for num_worker in num_workers:
            print(
                f"Running sequential benchmark with batch_size={batch_size},"
                f"num_workers={num_worker}, "
                f"prefetch_factor={prefetch_factor}"
            )
            results.append(
                run_sequential(
                    batch_size,
                    num_worker,
                    prefetch_factor,
                    paths,
                    args.test_time,
                )
            )
            save_results_to_csv(results, args.output_csv)


def main():
    parser = argparse.ArgumentParser(description="Benchmark DataLoader Performance")
    parser.add_argument("--path", type=str, default=None, help="Path of the anndata")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="dataloader_benchmark_results.csv",
        help="Path to save the benchmark results CSV",
    )
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--test_time", type=int, default=120, help="Duration to test each loader (in seconds)")
    args = parser.parse_args()

    if os.path.isfile(args.path):
        paths = [args.path]
    else:
        paths = os.listdir(args.path)
    paths = [os.path.join(args.path, p) for p in paths if p.endswith(".h5ad")]
    if args.mode == "seq":
        test_sequential(paths, args)
    elif args.mode == "random":
        test_random(paths, args)
    else:
        raise Exception("This mode is not supported")


if __name__ == "__main__":
    main()
