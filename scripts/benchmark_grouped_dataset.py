#!/usr/bin/env python3
"""
Standalone benchmarking script for GroupedPerturbIterableDataset performance.
This script measures cells per second throughput for different configurations.
"""

import sys
import os
import time
import argparse

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not available, progress bars will be disabled")
    def tqdm(iterable, *args, **kwargs):
        return iterable

from protoplast.scrna.anndata.pert_modules import PerturbDataModule, benchmark_dataloader


def run_benchmark_comparison():
    """Run comprehensive benchmarks comparing different configurations."""
    
    config_path = "notebooks/pert-dataconfig.toml"
    pert_embedding_file = "/home/tphan/Softwares/protoplast/notebooks/competition_support_set/ESM2_pert_features.pt"
    
    # Test configurations
    configs = [
        {
            "name": "Grouped S=4, B=8",
            "use_grouped_dataset": True,
            "group_size_S": 4,
            "train_batch_size": 8,
            "num_workers": 4,
        },
        {
            "name": "Grouped S=8, B=4", 
            "use_grouped_dataset": True,
            "group_size_S": 8,
            "train_batch_size": 4,
            "num_workers": 4,
        },
        {
            "name": "Grouped S=16, B=2",
            "use_grouped_dataset": True,
            "group_size_S": 16,
            "train_batch_size": 2,
            "num_workers": 4,
        },
        {
            "name": "Regular B=32",
            "use_grouped_dataset": False,
            "group_size_S": 1,  # Not used
            "train_batch_size": 32,
            "num_workers": 4,
        },
    ]
    
    results = {}
    
    print("="*60)
    print("COMPREHENSIVE DATALOADER BENCHMARK")
    print("="*60)
    
    for config in configs:
        print(f"\n{'='*20} {config['name']} {'='*20}")
        
        try:
            # Create data module
            dm = PerturbDataModule(
                config_path=config_path,
                pert_embedding_file=pert_embedding_file,
                use_grouped_dataset=config["use_grouped_dataset"],
                group_size_S=config["group_size_S"],
                train_batch_size=config["train_batch_size"],
                eval_batch_size=config["train_batch_size"],
                num_workers=config["num_workers"],
                persistent_workers=False,
                n_basal_samples=10,
                barcodes=False,
                seed=42
            )
            
            print("Setting up data module...")
            dm.setup()
            
            train_loader = dm.train_dataloader()
            
            # Benchmark
            result = benchmark_dataloader(
                train_loader, 
                config["name"], 
                max_batches=30  # Limit for quick comparison
            )
            
            results[config["name"]] = result
            
        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            results[config["name"]] = None
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Configuration':<25} {'Cells/sec':<12} {'Groups/sec':<12} {'Batches/sec':<12}")
    print("-" * 60)
    
    for name, result in results.items():
        if result:
            cells_per_sec = result['cells_per_second']
            groups_per_sec = result['total_groups'] / result['elapsed_time'] if result['elapsed_time'] > 0 else 0
            batches_per_sec = result['batches_per_second']
            print(f"{name:<25} {cells_per_sec:<12.1f} {groups_per_sec:<12.1f} {batches_per_sec:<12.1f}")
        else:
            print(f"{name:<25} {'FAILED':<12} {'-':<12} {'-':<12}")


def run_single_benchmark(group_size=8, batch_size=4, num_workers=4, max_batches=100):
    """Run a single benchmark with specified parameters."""
    
    config_path = "notebooks/pert-dataconfig.toml"
    pert_embedding_file = "/home/tphan/Softwares/protoplast/notebooks/competition_support_set/ESM2_pert_features.pt"
    
    print(f"Benchmarking GroupedPerturbIterableDataset:")
    print(f"  Group size: {group_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Max batches: {max_batches}")
    
    dm = PerturbDataModule(
        config_path=config_path,
        pert_embedding_file=pert_embedding_file,
        use_grouped_dataset=True,
        group_size_S=group_size,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
        n_basal_samples=10,
        barcodes=False,
        seed=42
    )
    
    print("Setting up data module...")
    dm.setup()
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    # Benchmark train and val
    train_result = benchmark_dataloader(train_loader, "Train", max_batches)
    val_result = benchmark_dataloader(val_loader, "Val", max_batches // 2)
    
    return train_result, val_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GroupedPerturbIterableDataset")
    parser.add_argument("--comparison", action="store_true", 
                       help="Run comparison benchmark across different configurations")
    parser.add_argument("--group-size", type=int, default=8,
                       help="Group size for single benchmark (default: 8)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for single benchmark (default: 4)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of workers (default: 4)")
    parser.add_argument("--max-batches", type=int, default=100,
                       help="Maximum batches to benchmark (default: 100)")
    
    args = parser.parse_args()
    
    if args.comparison:
        run_benchmark_comparison()
    else:
        run_single_benchmark(
            group_size=args.group_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches
        )
