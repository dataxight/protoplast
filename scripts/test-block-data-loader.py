import time
from tqdm import tqdm
from protoplast.scrna.anndata.torch_dataloader import BlockBasedAnnDataset
from torch.utils.data import DataLoader
import glob

def benchmark(loader, n_samples, batch_size):
    num_iter = n_samples // batch_size
    loader_iter = loader.__iter__()

    start_time = time.time()
    batch_times = []
    batch_time = time.time()
    for i, _batch in tqdm(enumerate(loader_iter), total=num_iter):
        batch_times.append(time.time() - batch_time)
        batch_time = time.time()
        if i == num_iter:
            break

    execution_time = time.time() - start_time
    time_per_sample = (1e6 * execution_time) / (num_iter * batch_size)
    print(f"time per sample: {time_per_sample:.2f} μs")
    samples_per_sec = num_iter * batch_size / execution_time
    print(f"samples per sec: {samples_per_sec:.2f} samples/sec")

    return samples_per_sec, time_per_sample, batch_times

def pass_through_collate_fn(batch):
    return batch[0]

def main():
    files = glob.glob("notebooks/competition_support_set/*.h5")
    ds = BlockBasedAnnDataset(
        file_paths=files,
        ds_batch_size=128,
        block_size=32,
        load_factor=32,
        sparse_keys=["X"]
    )
    dataloader = DataLoader(ds, batch_size=None, num_workers=16, prefetch_factor=16, collate_fn=pass_through_collate_fn, pin_memory=False, persistent_workers=False)
    benchmark(dataloader, ds.n_cells, 64)

if __name__ == "__main__":
    main()