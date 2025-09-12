import glob
import time
import argparse

import anndata as ad
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil
import os

from protoplast.scrna.anndata.torch_dataloader import DistributedAnnDataset, ann_split_data

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


def benchmark(loader, n_samples, batch_size, max_iteration=None, warmup_iteration=100):
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
        batch_times.append(time.time() - batch_time)
        batch_time = time.time()
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
    parser.add_argument("data_glob", type=str, help="glob pattern the h5 files")
    args = parser.parse_args()

    N_WORKERS = 32
    PREFETCH_FACTOR = 16
    # Example how to test throughput with DistributedAnnDataset
    files = glob.glob(args.data_glob)
    # files = glob.glob("/home/tphan/Softwares/protoplast/notebooks/competition_support_set/competition_train.h5")
    indices = ann_split_data(files, batch_size=64, test_size=0.0, validation_size=0.0)

    n_cells = 0

    for file in files:
        n_cells += ad.read_h5ad(file, backed="r").n_obs

    ds = DistributedAnnDataset(file_paths=files, indices=indices["train_indices"], sparse_keys=["X"], metadata={})
    dataloader = DataLoader(
        ds,
        batch_size=None,
        num_workers=N_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=False,
        persistent_workers=False,
    )
    samples_per_sec, time_per_sample, batch_times, peak_memory = benchmark(dataloader, n_cells, 64, max_iteration=10000)
    print(f"samples per sec: {samples_per_sec:.2f} samples/sec")
    print(f"time per sample: {time_per_sample:.2f} μs")
    print(f"peak memory: {peak_memory:.2f} MB")

if __name__ == "__main__":
    main()
