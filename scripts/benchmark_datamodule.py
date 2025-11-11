from dataclasses import dataclass
from protoplast.scrna.anndata.data_modules.perturbation import PerturbationDataModule
from cell_load.data_modules import PerturbationDataModule as CellLoadPerturbationDataModule

from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader

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
        num_workers=config.num_workers
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
            cells += batch[config.batch_item_key].shape[0]
    end_time = time()
    return cells / (end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-module", type=str, default="protoplast")
    args = parser.parse_args()

    config = BenchmarkConfig(
        data_config_path="/home/tphan/Softwares/vcc-models/configs/data-hq.toml",
        pert_embedding_file="/mnt/hdd2/tan/competition_support_set_sorted/ESM2_pert_features.pt",
        hvg_file="/home/tphan/Softwares/vcc-models/hvg-4000-competition-extended.txt",
        batch_size=16,
        num_workers=8,
        group_size_S=64,
        prefetch_factor=16,
        block_size=2048,
        batch_item_key="pert_cell_emb",
        basal_mapping_strategy="random",
        should_yield_control_cells=True,
    )
    if args.data_module == "protoplast":
        data_loader = create_protoplast_data_loader(config)
    elif args.data_module == "cellload":
        data_loader = create_cellload_data_loader(config)
    else:
        raise ValueError(f"Invalid data module: {args.data_module}")
    throughput = benchmark_datamodule(data_loader, config)
    print(f"Throughput: {throughput:.1f} cells/sec")