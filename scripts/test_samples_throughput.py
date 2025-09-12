import time
from tqdm import tqdm
from protoplast.scrna.anndata.torch_dataloader import DistributedAnnDataset, ann_split_data, cell_line_metadata_cb
from torch.utils.data import DataLoader
import glob
import anndata as ad
import tracemalloc

def benchmark(loader, n_samples, batch_size, max_iteration=None, warmup_iteration=100):
    tracemalloc.start()
    if max_iteration is None:
        # if no max_iteration is provided, we run for the entire dataset
        max_iteration = n_samples // batch_size
    loader_iter = loader.__iter__()

    start_time = time.time()
    batch_times = []
    batch_time = time.time()
    # we warmup for the first warmup_iteration iterations, then we run for max_iteration iterations
    max_iteration += warmup_iteration
    for i, _batch in tqdm(enumerate(loader_iter), total=max_iteration):
        batch_times.append(time.time() - batch_time)
        batch_time = time.time()
        #Assess peak memory
        if i == max_iteration:
            break

    _ , peak_memory = tracemalloc.get_traced_memory()
    peak_memory = peak_memory / 1024**2
    execution_time = time.time() - start_time
    time_per_sample = (1e6 * execution_time) / (max_iteration * batch_size)
    samples_per_sec = max_iteration * batch_size / execution_time
    tracemalloc.stop()

    return samples_per_sec, time_per_sample, batch_times, peak_memory

def pass_through_collate_fn(batch):
    return batch[0]

def main():
    files = glob.glob("/mnt/hdd2/tan/tahoe100m/plate7_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad")
    indices = ann_split_data(
        files, batch_size=64, test_size=0.0, validation_size=0.0
    ) 

    n_cells = 0

    for file in files:
        n_cells += ad.read_h5ad(file, backed="r").n_obs

    ds = DistributedAnnDataset(
        file_paths=files,
        indices=indices["train_indices"],
        sparse_keys=["X"],
        metadata={}
    )
    dataloader = DataLoader(ds, batch_size=None, num_workers=32, prefetch_factor=16, pin_memory=False, persistent_workers=False)
    samples_per_sec, time_per_sample, batch_times, peak_memory = benchmark(dataloader, n_cells, 64, max_iteration=10000)
    print(f"samples per sec: {samples_per_sec:.2f} samples/sec")
    print(f"time per sample: {time_per_sample:.2f} μs")
    print(f"peak memory: {peak_memory:.2f} MB")
    print(f"batch times: {batch_times}")

if __name__ == "__main__":
    main()
