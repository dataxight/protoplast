from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

import torch
from torch.utils.data import DataLoader, Subset

try:
    import pytorch_lightning as pl
except Exception:  # fallback stub
    class _PLStub:
        class LightningDataModule: ...
    pl = _PLStub()

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

# --- TOML loader: stdlib (3.11+) or fallback to 'toml' package ---
try:
    import tomllib as toml
except Exception:
    import toml  # type: ignore

from protoplast.scrna.anndata.pert_dataset import PerturbDataset, GroupedPerturbIterableDataset
from protoplast.scrna.train.utils import expand_globs 


@dataclass
class SplitIdxs:
    train: List[int]
    val: List[int]
    test: List[int]


class SplitPlanner:
    """
    Builds (train/val/test) index splits for a merged PerturbDataset
    using TOML config: [training], [zeroshot], [fewshot].
    """
    def __init__(
        self,
        cfg: dict,
        dataset: PerturbDataset,
        file_to_dataset_name: Dict[str, str],
    ):
        self.cfg = cfg
        self.ds = dataset
        self.file2name = file_to_dataset_name

        # Quick references to flattened arrays prepared in your dataset
        self.cell_types = self.ds.cell_types_flattened   # np.ndarray of str
        self.perturbs   = self.ds.perturbs_flattened     # np.ndarray of str
        self.n_cells    = int(self.ds.n_cells.sum())

        # precompute file spans to map global index -> file -> dataset name
        self.file_spans: List[Tuple[str, int, int]] = []
        offset = 0
        for f, n in zip(self.ds.h5ad_files, self.ds.n_cells):
            n = int(n)
            self.file_spans.append((f, offset, offset + n))
            offset += n

        # parse rules
        self.training_block: Dict[str, str] = {
            k: (v or "").lower() for k, v in self.cfg.get("training", {}).items()
        }
        self.zeroshot_map: Dict[str, str] = {
            k.lower(): (v or "").lower() for k, v in self.cfg.get("zeroshot", {}).items()
        }
        # fewshot: key "dataset.celltype" -> {'train': [...], 'val': [...], 'test': [...]}
        self.fewshot_map: Dict[str, Dict[str, List[str]]] = {}
        for k, v in self.cfg.get("fewshot", {}).items():
            self.fewshot_map[k.lower()] = {
                split: list(map(str, v.get(split, [])))
                for split in ("train", "val", "test")
                if split in v
            }

    def _dataset_name_for_index(self, idx: int) -> str:
        for f, s, e in self.file_spans:
            if s <= idx < e:
                return self.file2name[f]
        return "unknown"

    def plan(self) -> SplitIdxs:
        idxs = {"train": [], "val": [], "test": []}

        for i in range(self.n_cells):
            ds_name = self._dataset_name_for_index(i)
            ct = str(self.cell_types[i])
            pert = str(self.perturbs[i])
            key = f"{ds_name}.{ct}".lower()

            # 1) zeroshot override: whole cell type -> target split
            if key in self.zeroshot_map:
                idxs[self.zeroshot_map[key]].append(i)
                continue

            # 2) fewshot override: named perturbations -> specific split
            if key in self.fewshot_map:
                placed = False
                lists = self.fewshot_map[key]
                for split in ("train", "val", "test"):
                    if split in lists and pert in lists[split]:
                        idxs[split].append(i)
                        placed = True
                        break
                if placed:
                    continue
                # not listed: default to train
                idxs["train"].append(i)
                continue

            # 3) default: if dataset listed in [training] as "train"
            if self.training_block.get(ds_name, "") == "train":
                idxs["train"].append(i)
            else:
                # ignore (not part of training/val/test)
                pass

        return SplitIdxs(
            train=idxs["train"], val=idxs["val"], test=idxs["test"]
        )


def default_collate(batch_list):
    """
    Collate dicts returned by PerturbDataset.__getitem__.
    Stacks tensor values, keeps non-tensors as lists.
    """
    out = {}
    keys = batch_list[0].keys()
    for k in keys:
        vs = [b[k] for b in batch_list]
        if torch.is_tensor(vs[0]):
            out[k] = torch.stack(vs, dim=0)
        else:
            out[k] = vs
    return out


# ---------------------------
# The DataModule (factory)
# ---------------------------
class PerturbDataModule(pl.LightningDataModule):
    """
    DataModule that:
      - reads a TOML config
      - builds either a merged PerturbDataset or GroupedPerturbIterableDataset
      - applies zeroshot & fewshot split rules
      - exposes train/val/test DataLoaders
    
    When use_grouped_dataset=True, creates separate GroupedPerturbIterableDataset 
    instances for train/val/test that group cells by target_gene into 
    non-overlapping sets of size group_size_S. Each split respects the 
    zeroshot/fewshot configuration from the TOML file.
    """

    def __init__(
        self,
        config_path: str,
        pert_embedding_file: str,
        *,
        # dataset opts
        cell_type_label: str = "cell_type",
        target_label: str = "target_gene", 
        control_label: str = "non-targeting",
        batch_label: str = "batch_var", 
        use_batches: bool = True,
        n_basal_samples: int = 30,
        # grouped dataset opts
        use_grouped_dataset: bool = False,
        group_size_S: int = 8,
        seed: Optional[int] = None,
        # loader opts
        train_batch_size: int = 32,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = 4,
        collate_fn=default_collate,
        barcodes: bool = False
    ):
        super().__init__()
        self.config_path = config_path
        self.pert_embedding_file = pert_embedding_file

        # dataset opts
        self.cell_type_label = cell_type_label
        self.target_label = target_label
        self.control_label = control_label
        self.batch_label = batch_label
        self.use_batches = use_batches
        self.n_basal_samples = n_basal_samples
        
        # grouped dataset opts
        self.use_grouped_dataset = use_grouped_dataset
        self.group_size_S = group_size_S
        self.seed = seed

        # loader opts
        self.train_batch_size = int(train_batch_size)
        self.eval_batch_size = int(eval_batch_size or train_batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = prefetch_factor
        self._collate_fn = collate_fn
        self.barcodes = barcodes

        # populated in setup()
        self.cfg = None
        self.ds: Optional[PerturbDataset] = None
        self.train_subset = None
        self.val_subset = None
        self.test_subset = None

        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    @staticmethod
    def _load_toml(path: str) -> dict:
        with open(path, "rb") as f:
            return toml.load(f)

    @classmethod
    def from_toml(cls, config_path: str, pert_embedding_file: str):
        cfg = cls._load_toml(cls, config_path)  # use static parsing
        # pull defaults from toml if present
        lcfg = cfg.get("loader", {})
        dcfg = cfg.get("dataset_opts", {})
        gcfg = cfg.get("grouped_dataset", {})
        return cls(
            config_path=config_path,
            pert_embedding_file=pert_embedding_file,
            # dataset opts
            cell_type_label=str(dcfg.get("cell_type_label", "cell_type")),
            target_label=str(dcfg.get("target_label", "target_gene")),
            control_label=str(dcfg.get("control_label", "non-targeting")),
            batch_label=str(dcfg.get("batch_label", "batch_var")),
            use_batches=bool(dcfg.get("use_batches", True)),
            n_basal_samples=int(dcfg.get("n_basal_samples", 30)),
            # grouped dataset opts
            use_grouped_dataset=bool(gcfg.get("use_grouped_dataset", False)),
            group_size_S=int(gcfg.get("group_size_S", 8)),
            seed=gcfg.get("seed"),
            # loader opts
            train_batch_size=int(lcfg.get("batch_size", 32)),
            eval_batch_size=int(lcfg.get("eval_batch_size", lcfg.get("batch_size", 32))),
            num_workers=int(lcfg.get("num_workers", 4)),
            pin_memory=bool(lcfg.get("pin_memory", True)),
            persistent_workers=bool(lcfg.get("persistent_workers", True)),
            prefetch_factor=lcfg.get("prefetch_factor"),
        )

    # Lightning-style lifecycle ----------------------------------------------

    def prepare_data(self):
        # nothing to download; keep empty to be DDP-safe
        pass

    def setup(self, stage: Optional[str] = None):
        # 1) load config
        self.cfg = self._load_toml(self.config_path)

        # 2) collect files per dataset name via brace+glob
        datasets_block: Dict[str, str] = self.cfg.get("datasets", {})
        dataset_name_to_files: Dict[str, List[str]] = {}
        for name, pat in datasets_block.items():
            files = expand_globs(pat)
            if not files:
                raise FileNotFoundError(f"No files matched for datasets.{name} = {pat}")
            dataset_name_to_files[name] = files

        all_files: List[str] = []
        file_to_dataset_name: Dict[str, str] = {}
        for name, flist in dataset_name_to_files.items():
            for f in flist:
                all_files.append(f)
                file_to_dataset_name[f] = name

        # 3) build the dataset (for non-grouped case only)
        if not self.use_grouped_dataset:
            self.ds = PerturbDataset(
                h5ad_files=all_files,
                pert_embedding_file=self.pert_embedding_file,
                cell_type_label=self.cell_type_label,
                target_label=self.target_label,
                control_label=self.control_label,
                batch_label=self.batch_label,
                use_batches=self.use_batches,
                n_basal_samples=self.n_basal_samples,
                barcodes=self.barcodes
            )
        else:
            # For grouped datasets, we'll create them in the split section
            self.ds = None

        # 4) split planning and subset creation
        if self.use_grouped_dataset:
            # For GroupedPerturbIterableDataset, we need to first create a base PerturbDataset
            # to get the splits, then create separate GroupedPerturbIterableDataset instances
            base_dataset = PerturbDataset(
                h5ad_files=all_files,
                pert_embedding_file=self.pert_embedding_file,
                cell_type_label=self.cell_type_label,
                target_label=self.target_label,
                control_label=self.control_label,
                batch_label=self.batch_label,
                use_batches=self.use_batches,
                n_basal_samples=self.n_basal_samples,
                barcodes=self.barcodes
            )
            
            # Get split indices using the base dataset
            planner = SplitPlanner(
                cfg=self.cfg, dataset=base_dataset, file_to_dataset_name=file_to_dataset_name
            )
            split = planner.plan()
            
            # Create separate GroupedPerturbIterableDataset instances for each split
            self.train_subset = GroupedPerturbIterableDataset(
                base=base_dataset,
                group_size_S=self.group_size_S,
                valid_indices=split.train,
                seed=self.seed
            ) if split.train else None
            
            self.val_subset = GroupedPerturbIterableDataset(
                base=base_dataset,
                group_size_S=self.group_size_S,
                valid_indices=split.val,
                seed=self.seed
            ) if split.val else None
            
            self.test_subset = GroupedPerturbIterableDataset(
                base=base_dataset,
                group_size_S=self.group_size_S,
                valid_indices=split.test,
                seed=self.seed
            ) if split.test else None
            
            # Keep reference to the original full dataset (not used for training)
            self.ds = self.train_subset
        else:
            # For regular PerturbDataset, use the existing split planning
            planner = SplitPlanner(
                cfg=self.cfg, dataset=self.ds, file_to_dataset_name=file_to_dataset_name
            )
            split = planner.plan()

            # 5) build subsets
            self.train_subset = Subset(self.ds, split.train)
            self.val_subset   = Subset(self.ds, split.val) if split.val else None
            self.test_subset  = Subset(self.ds, split.test) if split.test else None

        # 6) build loaders (one-time)
        if self.use_grouped_dataset:
            # For GroupedPerturbIterableDataset, batch_size should be 1 since each item is already a group
            train_loader_kwargs = {
                "batch_size": self.train_batch_size,
                "shuffle": False,  # shuffling handled internally by the iterable dataset
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "persistent_workers": self.persistent_workers,
                "drop_last": False,
                "collate_fn": self._collate_fn,
            }
        else:
            # For regular PerturbDataset
            train_loader_kwargs = {
                "batch_size": self.train_batch_size,
                "shuffle": True,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "persistent_workers": self.persistent_workers,
                "drop_last": False,
                "collate_fn": self._collate_fn,
            }
        
        # Add prefetch_factor only if specified and num_workers > 0
        if self.prefetch_factor is not None and self.num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = self.prefetch_factor
            
        self._train_loader = DataLoader(self.train_subset, **train_loader_kwargs)

        def _mk_eval_loader(subset):
            if subset is None:
                return None
            if self.use_grouped_dataset:
                # For GroupedPerturbIterableDataset, batch_size should be 1
                return DataLoader(
                    subset,
                    batch_size=self.eval_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                    drop_last=False,
                    collate_fn=self._collate_fn,
                )
            else:
                # For regular PerturbDataset
                return DataLoader(
                    subset,
                    batch_size=self.eval_batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                    drop_last=False,
                    collate_fn=self._collate_fn,
                )

        self._val_loader  = _mk_eval_loader(self.val_subset)
        self._test_loader = _mk_eval_loader(self.test_subset)

    # Lightning hooks
    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader


def benchmark_dataloader(dataloader, name: str, max_batches: int = 100):
    """
    Benchmark a dataloader to measure cells per second throughput.
    
    Args:
        dataloader: PyTorch DataLoader to benchmark
        name: Name for the benchmark (used in output)
        max_batches: Maximum number of batches to process
    
    Returns:
        dict: Benchmark results including cells_per_second, total_cells, etc.
        
    Example:
        >>> dm = PerturbDataModule(config_path="config.toml", pert_embedding_file="emb.pt", use_grouped_dataset=True)
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
        >>> results = benchmark_dataloader(train_loader, "My Dataset", max_batches=50)
        >>> print(f"Throughput: {results['cells_per_second']:.1f} cells/sec")
    """
    print(f"\n=== Benchmarking {name} ===")
    
    if dataloader is None:
        print("Dataloader is None, skipping benchmark.")
        return
    
    total_cells = 0
    total_groups = 0
    start_time = time.time()
    
    # Use tqdm for progress tracking
    pbar = tqdm(enumerate(dataloader), total=min(max_batches, len(dataloader)), 
                desc=f"Benchmarking {name}", unit="batch")
    
    for batch_idx, batch in pbar:
        if batch_idx >= max_batches:
            break
            
        # Count cells in this batch
        if "pert_cell_emb" in batch:
            # For grouped datasets, shape is [batch_size, group_size, features]
            # For regular datasets, shape is [batch_size, features]
            cell_emb_shape = batch["pert_cell_emb"].shape
            if len(cell_emb_shape) == 3:  # Grouped dataset
                batch_cells = cell_emb_shape[0] * cell_emb_shape[1]  # batch_size * group_size
                total_groups += cell_emb_shape[0]
            else:  # Regular dataset
                batch_cells = cell_emb_shape[0]  # batch_size
            
            total_cells += batch_cells
            
            # Update progress bar with current rate
            elapsed = time.time() - start_time
            if elapsed > 0:
                cells_per_sec = total_cells / elapsed
                pbar.set_postfix({
                    'cells': total_cells,
                    'cells/sec': f'{cells_per_sec:.1f}',
                    'groups': total_groups if total_groups > 0 else 'N/A'
                })
    
    pbar.close()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if elapsed_time > 0:
        cells_per_second = total_cells / elapsed_time
        batches_per_second = (batch_idx + 1) / elapsed_time
        
        print(f"Results for {name}:")
        print(f"  Total cells processed: {total_cells:,}")
        print(f"  Total batches processed: {batch_idx + 1}")
        if total_groups > 0:
            print(f"  Total groups processed: {total_groups}")
            print(f"  Average cells per group: {total_cells / total_groups:.1f}")
        print(f"  Time elapsed: {elapsed_time:.2f} seconds")
        print(f"  Cells per second: {cells_per_second:.1f}")
        print(f"  Batches per second: {batches_per_second:.1f}")
    else:
        print(f"Benchmark completed too quickly to measure for {name}")
    
    return {
        'total_cells': total_cells,
        'total_batches': batch_idx + 1,
        'total_groups': total_groups,
        'elapsed_time': elapsed_time,
        'cells_per_second': cells_per_second if elapsed_time > 0 else 0,
        'batches_per_second': batches_per_second if elapsed_time > 0 else 0
    }

if __name__ == "__main__":
    # Test with regular PerturbDataset
    # print("Testing with regular PerturbDataset...")
    # dm = PerturbDataModule(
    #     config_path="notebooks/pert-dataconfig.toml",
    #     pert_embedding_file="/home/tphan/Softwares/protoplast/notebooks/competition_support_set/ESM2_pert_features.pt",
    #     train_batch_size=64,
    #     eval_batch_size=64,
    #     num_workers=8,
    #     persistent_workers=False,
    #     n_basal_samples=10,
    #     barcodes=True
    # )
    # dm.setup()
    # train_loader = dm.train_dataloader()
    # val_loader   = dm.val_dataloader()
    # test_loader  = dm.test_dataloader()
    # print("train loader size: ", len(train_loader))
    # print("val loader size: ", len(val_loader) if val_loader else "None")
    # for batch in train_loader:
    #     print("train (regular):")
    #     print(batch["pert_cell_emb"].shape)
    #     print(batch["cell_type_onehot"].shape)
    #     print(batch["pert_emb"].shape)
    #     print(batch["ctrl_cell_emb"].shape)
    #     print(batch["batch"].shape)
    #     break
    # if val_loader:
    #     for batch in val_loader:
    #         print("val (regular):")
    #         print(batch["pert_cell_emb"].shape)
    #         print(batch["cell_type_onehot"].shape)
    #         print(batch["pert_emb"].shape)
    #         print(batch["ctrl_cell_emb"].shape)
    #         print(batch["batch"].shape)
    #         break
    
    # Test with GroupedPerturbIterableDataset
    print("\nTesting with GroupedPerturbIterableDataset...")
    dm_grouped = PerturbDataModule(
        config_path="notebooks/pert-dataconfig.toml",
        pert_embedding_file="/home/tphan/Softwares/protoplast/notebooks/competition_support_set/ESM2_pert_features.pt",
        use_grouped_dataset=True,
        group_size_S=4,
        train_batch_size=64,  # Will be overridden to 1 for grouped
        eval_batch_size=64,
        num_workers=4,
        persistent_workers=False,
        n_basal_samples=10,
        barcodes=False,
        seed=42
    )
    dm_grouped.setup()
    train_loader_grouped = dm_grouped.train_dataloader()
    val_loader_grouped   = dm_grouped.val_dataloader()
    test_loader_grouped  = dm_grouped.test_dataloader()
    print("train loader size (grouped): ", len(train_loader_grouped) if train_loader_grouped else "None")
    print("val loader size (grouped): ", len(val_loader_grouped) if val_loader_grouped else "None")
    
    # Show sample batch shapes
    if train_loader_grouped:
        for batch in train_loader_grouped:
            print("Sample batch shapes (grouped train):")
            print(f"  pert_cell_emb: {batch['pert_cell_emb'].shape}")
            print(f"  cell_type_onehot: {batch['cell_type_onehot'].shape}")
            print(f"  pert_emb: {batch['pert_emb'].shape}")
            print(f"  ctrl_cell_emb: {batch['ctrl_cell_emb'].shape}")
            print(f"  batch: {batch['batch'].shape}")
            break
    
    # Benchmark the grouped dataloaders
    print("\n" + "="*50)
    print("BENCHMARKING GROUPED DATALOADERS")
    print("="*50)
    
    # Benchmark train loader
    train_results = benchmark_dataloader(
        train_loader_grouped, 
        "Grouped Train Loader", 
        max_batches=50  # Limit to 50 batches for quick benchmark
    )
    
    # Benchmark val loader
    val_results = benchmark_dataloader(
        val_loader_grouped, 
        "Grouped Val Loader", 
        max_batches=20  # Fewer batches for val
    )
    
    # Summary comparison
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    if train_results:
        print(f"Train Loader Performance:")
        print(f"  Cells/sec: {train_results['cells_per_second']:.1f}")
        print(f"  Groups/sec: {train_results['total_groups'] / train_results['elapsed_time']:.1f}")
    
    if val_results:
        print(f"Val Loader Performance:")
        print(f"  Cells/sec: {val_results['cells_per_second']:.1f}")
        print(f"  Groups/sec: {val_results['total_groups'] / val_results['elapsed_time']:.1f}")
    
    print("\nBenchmark completed!")