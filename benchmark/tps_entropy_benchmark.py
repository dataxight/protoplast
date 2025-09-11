import numpy as np
from tqdm import tqdm
import time
import gc
import pandas as pd
from scipy import stats
import argparse
from protoplast.scrna.anndata.trainer import DefaultShuffleStrategy
from protoplast.scrna.anndata.torch_dataloader import DistributedAnnDataset
from torch.utils.data import DataLoader
import os

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
    
    for i, batch in enumerate(loader):
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
                pbar.set_description(f"{description} (warming up: {current_time - warm_up_start:.1f}/{warm_up_seconds}s)")
                pbar.update(1)
                continue
        
        # Now we're past the warm-up period
        total_samples += batch_size
        
        elapsed = current_time - start_time
        pbar.set_postfix(samples=total_samples, elapsed=f"{elapsed:.2f}s")
        pbar.update(1)

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
    
    return {
        "samples_tested": total_samples,
        "elapsed": elapsed,
        "avg_time_per_sample": avg_time_per_sample,
        "samples_per_second": samples_per_second,
        "avg_batch_entropy": avg_batch_entropy,
        "std_batch_entropy": std_batch_entropy,
    }

def save_results_to_csv(results, filepath=None):
    """Save or update results to CSV file."""
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    if filepath is not None:
        df.to_csv(filepath, index=False)
        print(f"Updated results saved to {filepath}")
    
    return df

def run(batch_size, mini_batch_size, num_workers, prefetch_factor, paths, test_time):
    # Initialize shuffle strategy and split data
    shuffle_strategy = DefaultShuffleStrategy(
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
        **results
    }


def main():
    
    parser = argparse.ArgumentParser(description="Benchmark DataLoader Performance")
    parser.add_argument("--path", type=str, default=None, help="Path of the anndata")
    parser.add_argument("--output_csv", type=str, default="dataloader_benchmark_results.csv", help="Path to save the benchmark results CSV")
    parser.add_argument("--test_time", type=int, default=120, help="Duration to test each loader (in seconds)")
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        paths = [args.path]
    else:
        paths = os.listdir(args.path)
    paths = [os.path.join(args.path, p) for p in paths if p.endswith(".h5ad")]
    batch_size = [500, 1000, 2000]
    mini_batch_size = [100, 200, 250]
    num_workers = [4, 8, 16]
    prefetch_factor = 8
    results = []
    for batch_size in batch_size:
        for mini_batch_size in mini_batch_size:
            for num_workers in num_workers:
                print(f"Running benchmark with batch_size={batch_size}, mini_batch_size={mini_batch_size}, num_workers={num_workers}, prefetch_factor={prefetch_factor}")
                results.append(run(batch_size, mini_batch_size, num_workers, prefetch_factor, paths, args.test_time))
                save_results_to_csv(results, args.output_csv)

if __name__ == "__main__":
    main()
    
    