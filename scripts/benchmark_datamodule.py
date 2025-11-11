from dataclasses import dataclass
from protoplast.scrna.anndata.data_modules.perturbation import PerturbationDataModule
from cell_load.data_modules import PerturbationDataModule as CellLoadPerturbationDataModule

from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import itertools

import argparse

@dataclass
class BenchmarkConfig:
    hvg_file: str
    data_config_path: str
    pert_embedding_file: str
    batch_size: int
    num_workers: int
    group_size_S: int
    prefetch_factor: int
    block_size: int
    batch_item_key: str
    basal_mapping_strategy: str
    should_yield_control_cells: bool


def create_protoplast_data_loader(config: BenchmarkConfig):
    dm = PerturbationDataModule(
        config_path=config.data_config_path,
        hvg_file=config.hvg_file,
        pert_embedding_file=config.pert_embedding_file,
        group_size_S=config.group_size_S,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )
    dm.setup(stage="fit")
    return dm.train_dataloader()

def create_cellload_data_loader(config: BenchmarkConfig):
    dm = CellLoadPerturbationDataModule(
        toml_config_path=config.data_config_path,
        perturbation_features_file=config.pert_embedding_file,
        num_workers=config.num_workers,
        cell_sentence_len=config.group_size_S,
        basal_mapping_strategy=config.basal_mapping_strategy,
        should_yield_control_cells=config.should_yield_control_cells,
        batch_size=config.batch_size,
        batch_col="batch_var",
        pert_col="target_gene",
        cell_type_key="cell_type",
        control_pert="non-targeting"
    )
    dm.setup()
    return dm.train_dataloader()

def benchmark_datamodule(data_loader: DataLoader, config: BenchmarkConfig):
    cells = 0
    start_time = time()
    niter = len(data_loader)
    for batch in tqdm(data_loader, total=niter):
        if len(batch[config.batch_item_key].shape) == 2:
            # [B * S, G]
            cells += batch[config.batch_item_key].shape[0]
        elif len(batch[config.batch_item_key].shape) == 3:
            # [B, S, G]
            cells += batch[config.batch_item_key].shape[0] * batch[config.batch_item_key].shape[1]
    end_time = time()
    elapsed_time = end_time - start_time
    return cells / elapsed_time, elapsed_time, cells

def run_benchmark_matrix(
    data_module: str,
    batch_sizes: List[int],
    group_sizes: List[int],
    num_workers_list: List[int],
    base_config: BenchmarkConfig,
    output_file: str
):
    """Run benchmarks for all combinations of parameters and save to JSON."""
    results = {
        "data_module": data_module,
        "timestamp": datetime.now().isoformat(),
        "base_config": {
            "data_config_path": base_config.data_config_path,
            "pert_embedding_file": base_config.pert_embedding_file,
            "hvg_file": base_config.hvg_file,
            "basal_mapping_strategy": base_config.basal_mapping_strategy,
            "should_yield_control_cells": base_config.should_yield_control_cells,
        },
        "benchmarks": []
    }
    
    # Generate all combinations
    combinations = list(itertools.product(batch_sizes, group_sizes, num_workers_list))
    total_runs = len(combinations)
    
    print(f"\nRunning {total_runs} benchmark combinations...")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Group sizes: {group_sizes}")
    print(f"Num workers: {num_workers_list}")
    print(f"Data module: {data_module}\n")
    
    for idx, (batch_size, group_size, num_workers) in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"Run {idx}/{total_runs}: batch_size={batch_size}, group_size={group_size}, num_workers={num_workers}")
        print(f"{'='*80}")
        
        # Update config with current parameters
        config = BenchmarkConfig(
            data_config_path=base_config.data_config_path,
            pert_embedding_file=base_config.pert_embedding_file,
            hvg_file=base_config.hvg_file,
            batch_size=batch_size,
            num_workers=num_workers,
            group_size_S=group_size,
            prefetch_factor=base_config.prefetch_factor,
            block_size=base_config.block_size,
            batch_item_key=base_config.batch_item_key,
            basal_mapping_strategy=base_config.basal_mapping_strategy,
            should_yield_control_cells=base_config.should_yield_control_cells,
        )
        
        try:
            # Create data loader
            if data_module == "protoplast":
                data_loader = create_protoplast_data_loader(config)
            elif data_module == "cellload":
                data_loader = create_cellload_data_loader(config)
            else:
                raise ValueError(f"Invalid data module: {data_module}")
            
            # Run benchmark
            throughput, elapsed_time, total_cells = benchmark_datamodule(data_loader, config)
            
            # Store results
            result = {
                "batch_size": batch_size,
                "group_size": group_size,
                "num_workers": num_workers,
                "throughput_cells_per_sec": throughput,
                "elapsed_time_sec": elapsed_time,
                "total_cells": total_cells,
                "status": "success"
            }
            
            print(f"\n✓ Throughput: {throughput:.1f} cells/sec")
            print(f"  Total cells: {total_cells}")
            print(f"  Elapsed time: {elapsed_time:.2f} sec")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            result = {
                "batch_size": batch_size,
                "group_size": group_size,
                "num_workers": num_workers,
                "status": "failed",
                "error": str(e)
            }
        
        results["benchmarks"].append(result)
        
        # Save intermediate results after each run
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {output_file}")
    
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Results saved to {output_file}")
    print(f"{'='*80}\n")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark datamodule with various parameter combinations")
    parser.add_argument("--data-module", type=str, default="protoplast", 
                        choices=["protoplast", "cellload"],
                        help="Data module to benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128],
                        help="List of batch sizes to test (default: 16 32 64 128)")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[32, 64],
                        help="List of group sizes to test (default: 64 128)")
    parser.add_argument("--num-workers", type=int, nargs="+", default=[8, 16],
                        help="List of worker counts to test (default: 8 16)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: benchmark_results_{data_module}_{timestamp}.json)")
    parser.add_argument("--data-config-path", type=str, 
                        default="/home/tphan/Softwares/vcc-models/configs/data-benchmark.toml",
                        help="Path to data config TOML file")
    parser.add_argument("--pert-embedding-file", type=str,
                        default="/mnt/hdd2/tan/competition_support_set_sorted/ESM2_pert_features.pt",
                        help="Path to perturbation embedding file")
    parser.add_argument("--hvg-file", type=str,
                        default="/home/tphan/Softwares/vcc-models/hvg-4000-competition-extended.txt",
                        help="Path to HVG file")
    parser.add_argument("--basal-mapping-strategy", type=str, default="random",
                        help="Basal mapping strategy (for cellload)")
    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"benchmark_results_{args.data_module}_{timestamp}.json"
    
    # Create base config
    base_config = BenchmarkConfig(
        data_config_path=args.data_config_path,
        pert_embedding_file=args.pert_embedding_file,
        hvg_file=args.hvg_file,
        batch_size=16,  # Will be overridden in benchmark matrix
        num_workers=8,  # Will be overridden in benchmark matrix
        group_size_S=64,  # Will be overridden in benchmark matrix
        prefetch_factor=16,
        block_size=2048,
        batch_item_key="pert_cell_emb",
        basal_mapping_strategy=args.basal_mapping_strategy,
        should_yield_control_cells=True,
    )
    
    # Run benchmark matrix
    run_benchmark_matrix(
        data_module=args.data_module,
        batch_sizes=args.batch_sizes,
        group_sizes=args.group_sizes,
        num_workers_list=args.num_workers,
        base_config=base_config,
        output_file=args.output
    )