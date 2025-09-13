import time
from tqdm import tqdm
from protoplast.scrna.anndata.torch_dataloader import BlockBasedAnnDataset, DistributedAnnDataset, DistributedCellLineBlockBasedAnnDataset
from torch.utils.data import DataLoader
import glob
import math
from collections import Counter
import numpy as np
import torch 

MAX_ITERATION = 5000 # Number of batches iterated to benchmark


def compute_entropy(strings):
    total = len(strings)
    if total == 0:
        return 0.0

    counts = Counter(strings)
    entropy = 0.0

    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def benchmark(loader, n_samples, batch_size):
    num_iter = min(n_samples // batch_size, MAX_ITERATION)
    loader_iter = loader.__iter__()

    entropies = []
    for i, _batch in tqdm(enumerate(loader_iter), total=num_iter):
        all_plates = _batch
        entropies.append(compute_entropy(all_plates))
        
        if i == num_iter:
            break
    
    avg_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)
    print(f"average entropy: {avg_entropy:.2f}")
    print(f"std entropy: {std_entropy:.2f}")

    return avg_entropy, std_entropy


def benchmark_block_sampling_with_batch_fetch(dataset: list[str], ds_batch_size=64, block_size=32, load_factor=32, num_workers=32, prefetch_factor=16):
    def pass_through_collate_fn(batch):
        """
        Control what returned by the DataLoader as a batch.
        """
        return batch
    
    files = dataset
    ds = DistributedCellLineBlockBasedAnnDataset(
        file_paths=files,
        ds_batch_size=ds_batch_size,
        block_size=block_size,
        load_factor=load_factor,
        sparse_keys=["X"],
        obs_keys=["cell_line"]
    )
    dataloader = DataLoader(ds, batch_size=None, num_workers=num_workers, prefetch_factor=prefetch_factor, collate_fn=pass_through_collate_fn, pin_memory=False, persistent_workers=False)
    return benchmark(dataloader, ds.n_cells, 64)


if __name__ == "__main__":
    mean, sd = benchmark_block_sampling_with_batch_fetch(glob.glob("/mnt/hdd2/tan/tahoe100m/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"))