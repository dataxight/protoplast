import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import scplode as sp
import anndata as ad
import numpy as np
from collections import defaultdict
import logging
from typing import Optional, Sequence, Union, Dict, List, Tuple

from protoplast.scrna.train.utils import make_onehot_encoding_map

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler("perturb_dataset.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(stream_handler)


class PerturbDataset(Dataset):
    """
    PyTorch Dataset for perturbation scRNA-seq stored in AnnData h5ad files.

    Each sample: (x, y, b, xp)
        - x: normalized gene expression (float32 vector, length = #genes)
        - y: cell type label (int index)
        - b: batch label (int index, optional)
        - xp: perturbation target (int index, gene id or "control")
    """
    def __init__(
        self,
        h5ad_files: list[str],
        pert_embedding_file: str, # TODO: use onehot encoding if this file is not provided
        cell_type_label: str = "cell_type",
        target_label: str = "target_gene", 
        control_label: str = "non-targeting",
        batch_label: str = "batch_var", 
        use_batches: bool = True,
        n_basal_samples: int = 30,
        barcodes: bool = False
    ):
        self.control_label = control_label
        self.target_label = target_label
        self.cell_type_label = cell_type_label
        self.batch_label = batch_label
        self.use_batches = use_batches
        self.h5ad_files = h5ad_files
        self.sp_adatas = []
        self.pert_embedding_file = pert_embedding_file
        self.n_basal_samples = n_basal_samples
        self.barcodes = barcodes

        self.pert_embedding = torch.load(pert_embedding_file)

        # Load and concatenate AnnData objects
        for i,f in enumerate(h5ad_files):
            logger.info(f"write mmap file for {f}")
            adata = sp.read_h5ad(f)
            logger.info(f"n_obs for {f}: {adata.n_obs}")
            self.sp_adatas.append(adata)

        adatas = [ad.read_h5ad(f, backed="r") for f in h5ad_files]

        # get an array of the number of cells in each h5ad file
        self.n_cells = np.array([ad.n_obs for ad in adatas])
        logger.info(f"n_cells: {self.n_cells.sum()}")
        logger.info(f"n_genes: {self.sp_adatas[0].n_vars}")

        # get all cell barcodes across all h5ad files
        self.cell_barcodes_flattened = np.concatenate([ad.obs_names.tolist() for ad in adatas]).flatten()

        # get unique cell types across all h5ad files
        self.cell_types_flattened = np.concatenate([ad.obs[cell_type_label].tolist() for ad in adatas]).flatten()
        # Map categorical labels to integer ids
        self.cell_types_onehot_map = make_onehot_encoding_map(np.unique(self.cell_types_flattened))
        logger.info(f"Total unique cell types: {len(self.cell_types_onehot_map)}")

        self.perturbs_flattened = np.concatenate([ad.obs[target_label].tolist() for ad in adatas]).flatten()
        self.perturbs_identifiers = {p: i for i, p in enumerate(np.unique(self.perturbs_flattened))}
        # get unique batches across all h5ad files
        self.batches_flattened = np.concatenate([[f"f{i}_"] * ad.n_obs + ad.obs[batch_label].tolist() for i, ad in enumerate(adatas)]).flatten()
        self.batches_onehot_map = make_onehot_encoding_map(np.unique(self.batches_flattened))
        logger.info(f"Total unique batches: {len(self.batches_onehot_map)}")

        # Index controls by (y,b) for fast lookup
        self.control_index = np.where(self.perturbs_flattened == self.control_label)[0]
        logger.info(f"Total control cells: {len(self.control_index)}")
        self.control_lookup = defaultdict(list)
        # TODO: support batch strategy
        for i in self.control_index:
            self.control_lookup[self.cell_types_flattened[i]].append(i)

    def __len__(self):
        return self.n_cells.sum()

    def _get_file_idx(self, idx):
        # Get the index of the file that contains the cell
        return np.where(idx < self.n_cells.cumsum())[0][0]

    def get_onehot_cell_types(self, idx):
        cell_type = self.cell_types_flattened[idx]
        return self.cell_types_onehot_map[cell_type]

    def get_onehot_perturbs(self, idx):
        perturb = self.perturbs_flattened[idx]
        if perturb not in self.pert_embedding:
            # create all zero embedding
            self.pert_embedding[perturb] = torch.zeros(next(iter(self.pert_embedding.values())).shape[0])
        return self.pert_embedding[perturb]

    def get_onehot_batches(self, idx):
        batch = self.batches_flattened[idx]
        return self.batches_onehot_map[batch]
    
    def get_basal_samples(self, idx):
        # randomly sample n_basal_samples from the control_lookup
        # return in shape [K, G] where K is n_basal_samples
        cell_type = self.cell_types_flattened[idx]
        # collect control cells via "random" strategy
        # TODO: support batch strategy
        basal_samples_indices = np.random.choice(self.control_lookup[cell_type], size=self.n_basal_samples, replace=True)
        basal_samples_barcodes = self.cell_barcodes_flattened[basal_samples_indices]
        basal_samples = self.get_x_from_indices(basal_samples_indices)
        return basal_samples, basal_samples_barcodes

    def get_x_from_indices(self, indices):
        X = torch.tensor([], dtype=torch.float32)
        for idx in indices:
            file_idx = self._get_file_idx(idx)
            adata = self.sp_adatas[file_idx]
            barcode = self.cell_barcodes_flattened[idx]
            x = adata.get([barcode])
            x = torch.tensor(x, dtype=torch.float32)
            X = torch.cat([X, x])

        return X
    
    def __getitem__(self, idx):
        # Fetch expression row, convert sparse → dense → torch
        file_idx = self._get_file_idx(idx)
        adata = self.sp_adatas[file_idx]
        pert_barcode = self.cell_barcodes_flattened[idx]
        x = adata.get([pert_barcode])
        x = torch.tensor(x, dtype=torch.float32)
        y_onehot = self.get_onehot_cell_types(idx)
        b_onehot = self.get_onehot_batches(idx)
        xp_onehot = self.get_onehot_perturbs(idx)
        x_ctrl_matched, ctrl_barcodes = self.get_basal_samples(idx)
        pert_identifier = torch.tensor(self.perturbs_identifiers[self.perturbs_flattened[idx]], dtype=torch.int64)

        cell_type = self.cell_types_flattened[idx]

        sample = {
            "pert_cell_emb": x,
            "cell_type_onehot": y_onehot,
            "pert_emb": xp_onehot,
            "ctrl_cell_emb": x_ctrl_matched,
            "batch": b_onehot,
            "cell_type": cell_type,
            "pert_ident": pert_identifier
        }
        if self.barcodes:
            sample["pert_barcodes"] = pert_barcode
            sample["ctrl_barcodes"] = ctrl_barcodes
        return sample



class GroupedPerturbIterableDataset(IterableDataset):
    """
    IterableDataset that groups cells by target_gene into non-overlapping sets of size S.
    For each target, if its count isn't divisible by S, the last group is padded by
    sampling with replacement from that target's full pool.

    Each yielded item has an added leading S dimension:
      - pert_cell_emb:    [S, G]
      - cell_type_onehot: [S, N_cell_types]
      - pert_emb:         [S, D_pert]
      - ctrl_cell_emb:    [S, G]        (one matched control per S cell)
      - batch:            [S, N_batches]
      - cell_type:        list[str] length S
      - pert_ident:       [S] (int64)
      - (optional) pert_barcodes: list[str] length S
      - (optional) ctrl_barcodes: list[str] length S

    You can either pass an existing `PerturbDataset` via `base=` or let this class
    construct one by passing the same constructor args your dataset expects.
    
    Optionally, you can provide `valid_indices` to only include cells at those indices
    in the base dataset (useful for train/val/test splits).
    """

    def __init__(
        self,
        base: Union['PerturbDataset', None] = None,
        *,
        # If base is None, we construct a PerturbDataset using these:
        h5ad_files: Optional[Sequence[str]] = None,
        pert_embedding_file: Optional[str] = None,
        cell_type_label: str = "cell_type",
        target_label: str = "target_gene",
        control_label: str = "non-targeting",
        batch_label: str = "batch_var",
        use_batches: bool = True,
        n_basal_samples: int = 30,
        barcodes: bool = False,
        # Grouping controls:
        group_size_S: int = 8,
        # Split controls:
        valid_indices: Optional[Sequence[int]] = None,
        # Randomness:
        seed: Optional[int] = None,
    ):
        """
        Args:
            base: Existing PerturbDataset. If None, a new one is constructed from args below.
            h5ad_files, pert_embedding_file, ...: Passed to PerturbDataset if base is None.
            group_size_S: Group size S.
            valid_indices: Optional list of indices to include from base dataset (for splits).
            seed: Optional seed for reproducibility (affects group shuffling and padding choices).
        """
        if base is None:
            if h5ad_files is None or pert_embedding_file is None:
                raise ValueError("Provide either `base` or both `h5ad_files` and `pert_embedding_file`.")
            # Create PerturbDataset directly since we're in the same module
            base = PerturbDataset(
                h5ad_files=list(h5ad_files),
                pert_embedding_file=pert_embedding_file,
                cell_type_label=cell_type_label,
                target_label=target_label,
                control_label=control_label,
                batch_label=batch_label,
                use_batches=use_batches,
                n_basal_samples=n_basal_samples,
                barcodes=barcodes,
            )
        self.base = base
        
        # Store valid indices for filtering
        self.valid_indices = set(valid_indices) if valid_indices is not None else None

        self.S = int(group_size_S)
        if self.S <= 0:
            raise ValueError("group_size_S must be a positive integer.")

        self.seed = seed
        # Global RNG for group construction
        self._rng = np.random.default_rng(seed if seed is not None else np.random.SeedSequence().entropy)

        # Build: for each target label -> indices
        self.target_to_indices: Dict[str, np.ndarray] = self._build_target_index()

        # Precompute first-epoch groups and their count
        self.groups: List[np.ndarray] = self._build_groups_for_epoch()
        self._len = len(self.groups)

        logger.info(f"GroupedPerturbIterableDataset initialized: {self._len} groups (S={self.S}).")

    # ---------- Helpers ----------

    def _build_target_index(self) -> Dict[str, np.ndarray]:
        """Create mapping: target -> np.ndarray of indices in base dataset."""
        targets = np.unique(self.base.perturbs_flattened)
        t2idx: Dict[str, List[int]] = {t: [] for t in targets}
        for i, t in enumerate(self.base.perturbs_flattened):
            # Only include index if it's in valid_indices (if specified)
            if self.valid_indices is None or i in self.valid_indices:
                t2idx[t].append(i)
        
        # Remove empty targets (targets with no valid indices)
        t2idx = {t: idxs for t, idxs in t2idx.items() if idxs}
        t2idx_np = {t: np.asarray(idxs, dtype=np.int64) for t, idxs in t2idx.items()}

        # Log small preview
        preview = ", ".join([f"{t}:{len(v)}" for t, v in list(t2idx_np.items())[:10]])
        split_info = f" (filtered to {len(self.valid_indices)} indices)" if self.valid_indices else ""
        logger.info("Per-target counts (preview): " + (preview if preview else "no targets") + split_info)
        return t2idx_np

    def _build_groups_for_epoch(self) -> List[np.ndarray]:
        """
        For each target:
          - shuffle indices,
          - chunk into size S,
          - pad last chunk by sampling with replacement from that target if needed.
        Shuffle the final list of groups to randomize target order.
        """
        groups: List[np.ndarray] = []
        for t, idxs in self.target_to_indices.items():
            n = len(idxs)
            if n == 0:
                continue

            idxs_shuf = np.array(idxs, copy=True)
            self._rng.shuffle(idxs_shuf)

            full_chunks = n // self.S
            remainder = n % self.S

            # full chunks
            for c in range(full_chunks):
                start = c * self.S
                groups.append(idxs_shuf[start:start + self.S])

            # padded last chunk if needed
            if remainder > 0:
                tail = idxs_shuf[full_chunks * self.S :]
                pad_count = self.S - remainder
                pad = self._rng.choice(idxs, size=pad_count, replace=True)
                groups.append(np.concatenate([tail, pad], axis=0))

        self._rng.shuffle(groups)
        return groups

    def _iter_group_range_for_worker(self) -> Tuple[int, int]:
        """
        Split group list across workers (contiguous sharding).
        """
        n = len(self.groups)
        wi = get_worker_info()
        if wi is None:
            return 0, n
        per_worker = (n + wi.num_workers - 1) // wi.num_workers
        start = wi.id * per_worker
        end = min(start + per_worker, n)
        return start, end

    def _stack_onehots(self, getter_fn, idxs: np.ndarray) -> torch.Tensor:
        """
        Stack one-hot/embedding tensors returned by base getters over idxs into [S, ...].
        """
        items = []
        for i in idxs:
            t = getter_fn(int(i))
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            items.append(t)
        return torch.stack(items, dim=0)

    # ---------- PyTorch Dataset API ----------

    def __len__(self):
        # Number of S-sized groups in one epoch
        return self._len

    def __iter__(self):
        """
        Create a fresh epoch iterator:
          - rebuild (reshuffle) groups,
          - shard across workers,
          - for each group, fetch tensors and return a dict with S-leading dimension.
        """
        # Rebuild groups for a new epoch order
        self.groups = self._build_groups_for_epoch()
        start, end = self._iter_group_range_for_worker()

        # Local RNG if you want per-iterator randomness beyond group reshuffle
        local_rng = np.random.default_rng(self._rng.integers(0, 2**32 - 1))

        for g in range(start, end):
            idxs = self.groups[g]  # np.ndarray of length S

            # --- Perturbed expressions: [S, G]
            X = self.base.get_x_from_indices(idxs)

            # --- One-hots / embeddings
            Y = self._stack_onehots(self.base.get_onehot_cell_types, idxs)  # [S, N_cell_types]
            B = self._stack_onehots(self.base.get_onehot_batches, idxs)     # [S, N_batches]

            # pert embedding is the same across the group (same target); derive from the first index
            xp_vec = self.base.get_onehot_perturbs(int(idxs[0]))            # [D_pert] or [1, D_pert]
            if xp_vec.dim() == 1:
                XP = xp_vec.unsqueeze(0).repeat(self.S, 1)                  # [S, D_pert]
            else:
                XP = xp_vec.repeat_interleave(self.S, dim=0)                # defensive, keep [S, D_pert]

            # --- Controls: one matched control (same cell_type) per cell -> [S, G]
            ctrl_indices = []
            for i in idxs:
                ct = self.base.cell_types_flattened[int(i)]
                pool = self.base.control_lookup.get(ct, [])
                if len(pool) == 0:
                    # Fallback to any control if no same-type control exists (edge case)
                    pool = self.base.control_index
                j = local_rng.choice(pool, size=1, replace=True)[0]
                ctrl_indices.append(int(j))
            X_ctrl = self.base.get_x_from_indices(np.asarray(ctrl_indices, dtype=np.int64))  # [S, G]

            # --- Metadata
            cell_types = [self.base.cell_types_flattened[int(i)] for i in idxs]              # list[str], len S
            pert_ident = torch.stack([
                torch.tensor(self.base.perturbs_identifiers[self.base.perturbs_flattened[int(i)]],
                             dtype=torch.int64)
                for i in idxs
            ], dim=0)  # [S]

            sample = {
                "pert_cell_emb": X,            # [S, G]
                "cell_type_onehot": Y,         # [S, N_cell_types]
                "pert_emb": XP,                # [S, D_pert]
                "ctrl_cell_emb": X_ctrl,       # [S, G]
                "batch": B,                    # [S, N_batches]
                "cell_type": cell_types,       # list[str] length S
                "pert_ident": pert_ident,      # [S]
            }

            if self.base.barcodes:
                sample["pert_barcodes"] = self.base.cell_barcodes_flattened[idxs].tolist()  # [S]
                sample["ctrl_barcodes"] = self.base.cell_barcodes_flattened[
                    np.asarray(ctrl_indices, dtype=np.int64)
                ].tolist()  # [S]

            yield sample

if __name__ == "__main__":
    ds = GroupedPerturbIterableDataset(
        h5ad_files=["/home/tphan/Softwares/protoplast/notebooks/competition_support_set/competition_train.h5"],
        pert_embedding_file="/home/tphan/Softwares/protoplast/notebooks/competition_support_set/ESM2_pert_features.pt",
        barcodes=True
    )
    for di in ds:
        sample = di
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, len(v))
        break