from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset

try:
    import pytorch_lightning as pl
except Exception:  # fallback stub
    class _PLStub:
        class LightningDataModule: ...
    pl = _PLStub()

# --- TOML loader: stdlib (3.11+) or fallback to 'toml' package ---
try:
    import tomllib as toml
except Exception:
    import toml  # type: ignore

from protoplast.scrna.anndata.pert_dataset import PerturbDataset
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
      - builds a merged PerturbDataset (your class)
      - applies zeroshot & fewshot split rules
      - exposes train/val/test DataLoaders
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
        # loader opts
        train_batch_size: int = 32,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        collate_fn=default_collate,
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

        # loader opts
        self.train_batch_size = int(train_batch_size)
        self.eval_batch_size = int(eval_batch_size or train_batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self._collate_fn = collate_fn

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
            # loader opts
            train_batch_size=int(lcfg.get("batch_size", 32)),
            eval_batch_size=int(lcfg.get("eval_batch_size", lcfg.get("batch_size", 32))),
            num_workers=int(lcfg.get("num_workers", 4)),
            pin_memory=bool(lcfg.get("pin_memory", True)),
            persistent_workers=bool(lcfg.get("persistent_workers", True)),
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

        # 3) build the merged PerturbDataset
        self.ds = PerturbDataset(
            h5ad_files=all_files,
            pert_embedding_file=self.pert_embedding_file,
            cell_type_label=self.cell_type_label,
            target_label=self.target_label,
            control_label=self.control_label,
            batch_label=self.batch_label,
            use_batches=self.use_batches,
            n_basal_samples=self.n_basal_samples
        )

        # 4) split planning
        planner = SplitPlanner(
            cfg=self.cfg, dataset=self.ds, file_to_dataset_name=file_to_dataset_name
        )
        split = planner.plan()

        # 5) build subsets
        self.train_subset = Subset(self.ds, split.train)
        self.val_subset   = Subset(self.ds, split.val) if split.val else None
        self.test_subset  = Subset(self.ds, split.test) if split.test else None

        # 6) build loaders (one-time)
        self._train_loader = DataLoader(
            self.train_subset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self._collate_fn,
        )

        def _mk_eval_loader(subset):
            if subset is None:
                return None
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

        self._val_loader  = _mk_eval_loader(self.val_subset)
        self._test_loader = _mk_eval_loader(self.test_subset)

    # Lightning hooks
    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader

if __name__ == "__main__":
    dm = PerturbDataModule(
        config_path="notebooks/pert-dataconfig.toml",
        pert_embedding_file="/home/tphan/Softwares/protoplast/notebooks/competition_support_set/ESM2_pert_features.pt",
        train_batch_size=64,
        eval_batch_size=64,
        num_workers=8,
        persistent_workers=False,
        n_basal_samples=10
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
    test_loader  = dm.test_dataloader()
    print("train loader size: ", len(train_loader))
    print("val loader size: ", len(val_loader))
    for batch in train_loader:
        print("train")
        print(batch["pert_cell_emb"].shape)
        print(batch["cell_type_onehot"].shape)
        print(batch["pert_emb"].shape)
        print(batch["ctrl_cell_emb"].shape)
        print(batch["batch"].shape)
        break
    for batch in val_loader:
        print("val")
        print(batch["pert_cell_emb"].shape)
        print(batch["cell_type_onehot"].shape)
        print(batch["pert_emb"].shape)
        print(batch["ctrl_cell_emb"].shape)
        print(batch["batch"].shape)
        break