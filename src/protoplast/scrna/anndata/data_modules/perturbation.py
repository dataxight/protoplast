import logging
from copy import deepcopy
from pathlib import Path
import os
import pickle

import anndata
import h5py
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import tqdm
import torch.distributed as td
from scipy.sparse import csr_matrix

from anndata._core.sparse_dataset import sparse_dataset
from line_profiler import profile
from torch.utils.data import get_worker_info
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protoplast.scrna.anndata.data_modules.utils import make_onehot_encoding_map, parse_dataset_config, collate_sparse_matrices_torch_direct
from protoplast.scrna.anndata.torch_dataloader import AnnDataModule, DistributedAnnDataset

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('perturbation.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

def slice_csr_rows(A, row_start, row_end):
    row_end = min(row_end, A.shape[0])
    start_ptr, end_ptr = A.indptr[row_start], A.indptr[row_end]
    data = A.data[start_ptr:end_ptr]
    indices = A.indices[start_ptr:end_ptr]
    indptr = A.indptr[row_start:row_end+1] - start_ptr
    return csr_matrix((data, indices, indptr), shape=(row_end - row_start, A.shape[1]))

class PerturbationDataset(DistributedAnnDataset):
    """
    PyTorch Dataset for perturbation scRNA-seq stored in AnnData h5ad files.

    The h5ad files are pre-sorted by target-gene, each file contains
    cells from only one cell type.

    Each sample contains:
        - pert_cell_emb: tensor sparse csr matrix [S, G] for perturbation cells
        - ctrl_cell_emb: tensor sparse csr matrix [S, G] for control cells
        - pert_emb: tensor [5102] perturbation embedding
        - pert_name: perturbation (target_gene) name
        - cell_type_onehot: tensor [n_cell_type] cell type
        - batch_onehot: tensor [n_batches] batch information
        - pert_cell_barcode: np.ndarray [S] cell barcodes (if barcode=True)
        - ctrl_cell_barcode: np.ndarray [S] control cell barcodes (if barcode=True)
    """
    def __init__(
        self,
        pert_embedding_file: str,
        hvg_file: str = None,
        cell_type_label: str = "cell_type",
        target_label: str = "target_gene",
        control_label: str = "non-targeting",
        batch_label: str = "batch_var",
        use_batches: bool = True,
        group_size_S: int = 32,
        barcodes: bool = False,
        n_items: int = None,
        cell_noise: float = 0.3,
        gene_noise: float = 0.3,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.control_label = control_label
        self.target_label = target_label
        self.cell_type_label = cell_type_label
        self.batch_label = batch_label
        self.use_batches = use_batches
        self.pert_embedding_file = pert_embedding_file
        self.hvg_file = hvg_file
        self.group_size_S = group_size_S
        self.barcodes = barcodes
        self.n_items = n_items
        self.h5files = None
        self.adata_obs = None
        self.cell_noise = cell_noise
        self.gene_noise = gene_noise
        self.adatas = None
        self.pert_embedding = torch.load(pert_embedding_file)

        # Initialize control region tracking per cell type
        self.cell_type_ctrl_regions = {}  # {cell_type: (start, end)} per file
        self.cell_type_ctrl_mat = {}  # {cell_type: scipy csr matrix} per file

        self._initialize_region_mappings()
        self._initialized_worker_info = False

    @profile
    def _get_mat_by_range(self, h5file: h5py.File, start: int, end: int):
        """Get matrix by range."""
        if len(self.sparse_keys) == 1:
            # mat = getattr(adata, self.sparse_keys[0])[start:end]
            mat = sparse_dataset(h5file[self.sparse_keys[0]])
            mat = mat[start:end]
            if not scipy.sparse.issparse(mat):
                mat = scipy.sparse.csr_matrix(mat)
            return mat
        else:
            raise ValueError(f"Multiple sparse keys are not supported for perturbation dataset: {self.sparse_keys}")

    def _init_worker_info(self):
        self._initialized_worker_info = True
        worker_info = get_worker_info()
        if worker_info is None:
            self.wid = 0
            self.nworkers = 1
        else:
            self.wid = worker_info.id
            self.nworkers = worker_info.num_workers
        try:
            w_rank = td.get_rank()
            w_size = td.get_world_size()
        except ValueError:
            w_rank = -1
            w_size = -1
        if w_rank >= 0:
            self.ray_rank = w_rank
            self.ray_size = w_size
        else:
            self.ray_rank = 0
            self.ray_size = 1

    @profile
    def _initialize_region_mappings(self):
        """Initialize region mappings and control regions per cell type."""
        # Load adatas to analyze structure
        adatas = [anndata.read_h5ad(f, backed="r") for f in self.files]

        logger.info(f"n_cells: {np.array([ad.n_obs for ad in adatas]).sum()}")
        logger.info(f"n_genes: {adatas[0].n_vars}")

        # Build onehot mappings
        cell_types_flattened = np.concatenate([ad.obs[self.cell_type_label].tolist() for ad in adatas]).flatten()
        self.cell_types_onehot_map = make_onehot_encoding_map(np.unique(cell_types_flattened))
        logger.info(f"Total unique cell types: {len(self.cell_types_onehot_map)}")

        # Build onehot mapping for perturbations
        pert_names_flattened = np.concatenate([ad.obs[self.target_label].tolist() for ad in adatas]).flatten()
        unique_pert_names = sorted(np.unique(pert_names_flattened))
        self.pert_names_map = {name: i for i, name in enumerate(unique_pert_names)}
        logger.info(f"Total unique perturbation names: {len(self.pert_names_map)}")

        batches_flattened = np.concatenate([
            [f"f{i}_"] * ad.n_obs + ad.obs[self.batch_label].tolist() for i, ad in enumerate(adatas)
        ]).flatten()
        self.batches_onehot_map = make_onehot_encoding_map(np.unique(batches_flattened))
        logger.info(f"Total unique batches: {len(self.batches_onehot_map)}")

        # Track control regions for each cell type (each file has one cell type)
        for file_i, adata in enumerate(adatas):
            cell_type = adata.obs[self.cell_type_label].iloc[0]

            # Find control regions in this file
            ctrl_mask = adata.obs[self.target_label] == self.control_label
            if ctrl_mask.any():
                ctrl_indices = np.where(ctrl_mask)[0]
                ctrl_start, ctrl_end = ctrl_indices[0], ctrl_indices[-1] + 1

                # logger.info(f"Prefetch control matrix for cell type {cell_type} in file {file_i}")
                if cell_type not in self.cell_type_ctrl_regions:
                    self.cell_type_ctrl_regions[cell_type] = {}
                    # self.cell_type_ctrl_mat[cell_type] = {}
                self.cell_type_ctrl_regions[cell_type][file_i] = (ctrl_start, ctrl_end)
                # self.cell_type_ctrl_mat[cell_type][file_i] = adata.X[ctrl_start:ctrl_end]

        logger.info(f"Control regions per cell type: {dict(self.cell_type_ctrl_regions)}")

    @classmethod
    def create_distributed_ds(cls, indices: dict, sparse_keys: list[str], mode: str = "train", **kwargs):
        """
        Create distributed dataset with target regions.

        indices format:
        {
            "files": [list of file paths],
            "train_indices": [...target_regions],  # List of (file_i, start_i, end_i)
            "metadata": {},
            "sparse_keys": ["X"]
        }
        """
        target_regions = indices[f"{mode}_indices"]

        return cls(file_paths=indices["files"],
                    indices=target_regions,
                    metadata=indices["metadata"],
                    n_items=indices[f"{mode}_n_items"],
                    sparse_keys=sparse_keys, **kwargs)

    def _get_pert_embedding(self, pert_id: str):
        """Get perturbation embedding, create zero embedding if not found."""
        if pert_id not in self.pert_embedding:
            # create all zero embedding
            self.pert_embedding[pert_id] = torch.zeros(next(iter(self.pert_embedding.values())).shape[0])
        return self.pert_embedding[pert_id]

    def get_pert_name_idx(self, pert_name: str):
        """Get index for perturbation name."""
        return self.pert_names_map[pert_name]

    def get_n_perts(self):
        """Get number of perturbations."""
        return len(self.pert_names_map)

    def get_celltype_onehot(self, cell_type: str):
        """Get onehot encoding for cell type."""
        return self.cell_types_onehot_map[cell_type]

    def get_batch_onehot(self, batch: str):
        """Get onehot encoding for batch."""
        return self.batches_onehot_map[batch]

    @profile
    def _produce_data(self, file_i: int, start: int, end: int):
        """
        Produce perturbation data from a region by splitting into target-specific groups.

        Args:
            file_i: File index
            start: Region start index
            end: Region end index

        Yields:
            Tuple of (X_pert, cell_indices) for each target group
        """
        adata_obs = self.adata_obs[file_i]
        X = self._get_mat_by_range(self.h5files[file_i], start, end)

        # Get targets in this region
        targets = adata_obs[self.target_label][start:end].values.tolist()
        target_counts = pd.Series(targets).value_counts(sort=False)
        target_cumsum = target_counts.cumsum()
        target_cumsum = np.insert(target_cumsum.values, 0, 0)
        # print(start, end, target_counts, target_cumsum, targets, np.unique(targets))

        for i in range(1, len(target_counts) + 1):
            start_i, end_i = target_cumsum[i-1], target_cumsum[i]
            target = targets[start_i]

            for j in range(start_i, end_i, self.group_size_S):
                # X_pert_group = X_pert[j:j+self.group_size_S]
                n_items = min(end_i - j, self.group_size_S)
                X_pert_group = X[j:j+n_items]
                cell_indices_group = np.arange(j + start, j + n_items + start)
                assert X_pert_group.shape[0] == n_items, f"X_pert shape mismatch: {X_pert_group.shape[0]} != {n_items}"
                assert len(cell_indices_group) == n_items, f"Cell indices length mismatch: {len(cell_indices_group)} != {n_items}, start: {start}, end: {end}, start_i: {start_i}, end_i: {end_i}"

                if X_pert_group.shape[0] < self.group_size_S:
                    sample_indices = np.random.choice(X_pert_group.shape[0], self.group_size_S, replace=True)
                    X_pert_group = X_pert_group[sample_indices]
                    cell_indices_group = cell_indices_group[sample_indices]
                    assert X_pert_group.shape[0] == self.group_size_S, f"X_pert shape mismatch: {X_pert_group.shape[0]} != {self.group_size_S}"
                    assert len(cell_indices_group) == self.group_size_S, f"Cell indices length mismatch: {len(cell_indices_group)} != {self.group_size_S}, {cell_indices_group}"

                yield X_pert_group, cell_indices_group, target
    @profile
    def sampling_control(self, cell_type: str, file_i: int, target_number: int):
        """
        Sample control cells matching covariate cell type and batch.

        Args:
            cell_type: Cell type to match
            file_i: File index
            target_number: Number of cells to sample

        Returns:
            Sampled control cell matrix and barcodes (if enabled)
        """
        if cell_type not in self.cell_type_ctrl_regions:
            raise ValueError(f"No control regions found for cell type: {cell_type}")

        if file_i not in self.cell_type_ctrl_regions[cell_type]:
            raise ValueError(f"File index {file_i} not found in control regions for cell type {cell_type}")

        start, end = self.cell_type_ctrl_regions[cell_type][file_i]
        # mat = self.cell_type_ctrl_mat[cell_type][file_i]

        if (end - start) > target_number:
            start_pos = start + np.random.randint(0, end - start - target_number + 1)
            end_pos = start_pos + target_number
            X = self._get_mat_by_range(self.h5files[file_i], start_pos, end_pos)
            # X = mat[start_pos:end_pos]
            barcodes = self.adata_obs[file_i].index[start_pos:end_pos].values
        else:
            X = self._get_mat_by_range(self.h5files[file_i], start, end)
            # X = mat[start:end]
            barcodes = self.adata_obs[file_i].index[start:end].values
        # if not enough cells, sample with replacement
        if X.shape[0] < target_number:
            indices = np.random.choice(X.shape[0], target_number, replace=True)
            X = X[indices]
            barcodes = barcodes[indices]

        return X, barcodes

    def get_hvg_cos_dis(self, gene, hvg_genes):
        gene_emb = self._get_pert_embedding(gene)
        hvg_gene_emb = [self._get_pert_embedding(gene) for gene in hvg_genes]
        hvg_gene_emb = torch.stack(hvg_gene_emb) # tensor [E, 5120]
        hvg_gene_cos_sim = F.cosine_similarity(hvg_gene_emb, gene_emb.unsqueeze(0), dim=1) # tensor [E]
        hvg_gene_cos_dis = 1 - hvg_gene_cos_sim
        return hvg_gene_cos_dis

    def __len__(self):
        return self.n_items

    def __iter__(self):
        """Iterate over perturbation samples."""
        if not self._initialized_worker_info:
            self._init_worker_info()
        # Load anndata objects in backed mode
        if self.adata_obs is None:
            logger.info(f"Loading anndata objects in backed mode")
            self.adata_obs = []
            self.adata_vars = []
            for f in self.files:
                adata = anndata.read_h5ad(f, backed="r")
                self.adata_obs.append(deepcopy(adata.obs))
                self.adata_vars.append(deepcopy(adata.var))
            if self._hvg_genes is not None:
                self._hvg_mask = np.where(self.adata_vars[0].index.isin(self._hvg_genes))[0]
            logger.info(f"Anndata objects loaded")
        if self.h5files is None:
            logger.info(f"Loading h5 files in swmr mode")
            self.h5files = [h5py.File(f, 'r', libver='latest', swmr=True) for f in self.files]
            logger.info(f"H5 files loaded")

        import random
        random.shuffle(self.batches)
        for region_idx, region in enumerate(self.batches):
            file_i, start, end = region

            # Worker/rank filtering
            if not (region_idx % self.ray_size == self.ray_rank and region_idx % self.nworkers == self.wid):
                continue

            try:
                # Get cell type (each file is one cell type only)
                cell_type = self.adata_obs[file_i][self.cell_type_label].iloc[0]

                # Process each target in this region
                for X_pert, cell_indices, target in self._produce_data(file_i, start, end):
                    # Get batches for perturbation cells
                    batches = self.adata_obs[file_i][self.batch_label].iloc[cell_indices].values

                    # Get pert embedding
                    # X_pert_emb = self.adatas[file_i].obsm["X_emb"][cell_indices]

                    # Get control cells with matching covariates
                    X_ctrl, ctrl_barcodes = self.sampling_control(cell_type, file_i, self.group_size_S)
                    # ctrl_indices = np.where(self.adata_obs[file_i].index.isin(ctrl_barcodes))[0]
                    # X_ctrl_emb = self.adatas[file_i].obsm["X_emb"][ctrl_indices]

                    # Get embeddings and onehot encodings
                    pert_emb = self._get_pert_embedding(target)
                    cell_type_onehot = self.get_celltype_onehot(cell_type)
                    batch_onehots = [self.get_batch_onehot(batch) for batch in batches]
                    pert_idx = self.get_pert_name_idx(target)

                    # Get barcodes for perturbation cells if needed
                    pert_barcodes = None
                    if self.barcodes:
                        pert_barcodes = self.adata_obs[file_i].index[cell_indices].values

                    if self.hvg_only:
                        # hvg_mask = np.where(np.array(self.adata_vars[file_i]["highly_variable"]))[0]
                        X_pert_emb = X_pert[:, self._hvg_mask]
                        X_ctrl_emb = X_ctrl[:, self._hvg_mask]
                    else:
                        X_pert_emb = X_pert
                        X_ctrl_emb = X_ctrl

                    hvg_genes = self.adata_vars[file_i].index[self._hvg_mask]
                    # hvg_gene_cos_dis = self.get_hvg_cos_dis(target, hvg_genes)
                    # Create sample dictionary
                    sample = {
                        # "pert_cell_g": X_pert.astype(np.float32), # scipy csr matrix [S, G]
                        # "ctrl_cell_g": X_ctrl.astype(np.float32), # scipy csr matrix [S, G]
                        "pert_cell_emb": X_pert_emb.astype(np.float32), # tensor [S, E]
                        "ctrl_cell_emb": X_ctrl_emb.astype(np.float32), # tensor [S, E]
                        "pert_emb": pert_emb, # tensor [5102]
                        "pert_name": np.array([target]),
                        "cell_type": np.array([cell_type]), # str
                        "cell_type_onehot": torch.stack([cell_type_onehot] * self.group_size_S), # [S, n_cell_type]
                        "batch_onehot": torch.stack(batch_onehots), # tensor [S, n_batches]
                        "pert_idx": torch.tensor([pert_idx], dtype=torch.long), # tensor [n_pert_names]
                    }

                    # Add barcodes if enabled
                    if self.barcodes:
                        sample["pert_cell_barcode"] = pert_barcodes # np.ndarray [S]
                        sample["ctrl_cell_barcode"] = ctrl_barcodes # np.ndarray [S]

                    yield sample

            except Exception as e:
                # logger.warning(f"Error processing region {region}: {e}")
                raise e


class PerturbationDataModule(AnnDataModule):
    """
    PyTorch Lightning DataModule for perturbation scRNA-seq data.

    Subclasses AnnDataModule to work with PerturbationDataset.

    Usage Examples:

    1. Using config file:
        dm = PerturbationDataModule(
            config_path="path/to/config.toml",
            pert_embedding_file="path/to/embeddings.pt"
        )

    2. Manual file specification:
        dm = PerturbationDataModule(
            files=["file1.h5", "file2.h5"],
            pert_embedding_file="path/to/embeddings.pt",
            cell_type_label="cell_type",
            target_label="target_gene"
        )

    The config file should follow this structure:
        [datasets]
        dataset_name = "/path/to/files/{file1,file2,file3}.h5"

        [zeroshot]
        "dataset_name.cell_type1" = "test"
        "dataset_name.cell_type2" = "test"

        [dataset_opts]
        cell_type_label = "cell_type"
        target_label = "target_gene"
        control_label = "non-targeting"
        batch_label = "batch_var"
        use_batches = true
        n_basal_samples = 32
    """
    def __init__(
        self,
        files: list[str] = None,
        pert_embedding_file: str = None,
        hvg_file: str = None,
        config_path: str = None,
        num_workers: int = None,
        prefetch_factor: int = 2,
        cell_type_label: str = "cell_type",
        target_label: str = "target_gene",
        control_label: str = "non-targeting",
        batch_label: str = "batch_var",
        use_batches: bool = True,
        group_size_S: int = 32,
        barcodes: bool = False,
        block_size: int = 2048,
        batch_size: int = 64,
        **kwargs
    ):
        # Handle config-based initialization
        if config_path:
            config_data = parse_dataset_config(config_path)
            file_splits = config_data["file_splits"]
            target_splits = config_data["target_splits"]
            config = config_data["config"]

            # Extract all files from config
            if files is None:
                files = list(file_splits.keys())

            # Override parameters from config if available
            if "dataset_opts" in config:
                dataset_opts = config["dataset_opts"]
                cell_type_label = dataset_opts.get("cell_type_label", cell_type_label)
                target_label = dataset_opts.get("target_label", target_label)
                control_label = dataset_opts.get("control_label", control_label)
                batch_label = dataset_opts.get("batch_label", batch_label)
                use_batches = dataset_opts.get("use_batches", use_batches)
                group_size_S = dataset_opts.get("n_basal_samples", group_size_S)

            if "loader" in config:
                loader_opts = config["loader"]
                num_workers = loader_opts.get("num_workers", num_workers)
                prefetch_factor = loader_opts.get("prefetch_factor", prefetch_factor)
        else:
            file_splits = None
            target_splits = None

        if not kwargs.get("sparse_keys", None):
            kwargs["sparse_keys"] = ["X"]

        indices = self.build_indices(files,
                                    target_label,
                                    control_label,
                                    block_size,
                                    group_size_S,
                                    file_splits,
                                    target_splits
                                )
        # Initialize parent with PerturbationDataset
        super().__init__(
            indices=indices,
            dataset=PerturbationDataset,
            prefetch_factor=prefetch_factor,
            **kwargs
        )
        # shuffle batches

        if num_workers is not None:
            self.loader_config["num_workers"] = num_workers

        # set collate fn
        self.loader_config["collate_fn"] = self.collate_fn
        self.loader_config["batch_size"] = batch_size

        # Store perturbation-specific parameters
        self.pert_embedding_file = pert_embedding_file
        self.hvg_file = hvg_file
        self.cell_type_label = cell_type_label
        self.target_label = target_label
        self.control_label = control_label
        self.batch_label = batch_label
        self.use_batches = use_batches
        self.group_size_S = group_size_S
        self.barcodes = barcodes

    @staticmethod
    def build_indices(files: list[str],
                      target_label: str = "target_gene",
                      control_label: str = "non-targeting",
                      block_size: int = 2048,
                      group_size_S: int = 32,
                      file_splits: dict = None,
                      target_splits: dict = None):
        """Build indices for perturbation dataset with fewshot support."""
        indices = {
            "files": files,
            "train_indices": [],
            "val_indices": [],
            "test_indices": [],
            "metadata": {},
        }

        n_items = {"train": 0, "val": 0, "test": 0}

        for file_i, file in enumerate(files):
            adata = anndata.read_h5ad(file, backed="r")

            # Get file metadata
            if file_splits and file in file_splits:
                file_info = file_splits[file]
                file_split = file_info["split"]
                dataset_name = file_info["dataset"]
                cell_type = file_info["cell_type"]
            else:
                file_split = "train"  # default
                dataset_name = "unknown"
                cell_type = Path(file).stem

            # Get target-specific splits for this file
            target_split_map = {}
            if target_splits and (dataset_name, cell_type) in target_splits:
                target_split_map = target_splits[(dataset_name, cell_type)]

            # Create regions that respect target boundaries and minimum block size
            target_counts = adata.obs.groupby(target_label, observed=True).agg({target_label: "count"})
            cumsum = np.cumsum(target_counts[target_label])
            cumsum = np.insert(cumsum, 0, 0)

            # Track regions by split
            split_regions = {"train": [], "val": [], "test": []}

            # Process each target individually for fewshot support
            for i in range(1, len(target_counts) + 1):
                target_start = cumsum[i-1]
                target_end = cumsum[i]
                current_target = target_counts.index[i-1]

                if current_target == control_label:
                    continue  # Skip control targets for now

                # Determine split for this target
                if current_target in target_split_map:
                    target_split = target_split_map[current_target]
                else:
                    target_split = file_split  # Use file's default split

                # Count items for this target
                target_size = target_end - target_start
                items = target_size // group_size_S
                if items == 0:
                    items = 1
                n_items[target_split] += items

                # Add target region to appropriate split
                split_regions[target_split].append((target_start, target_end))

            # Create consolidated regions for each split, respecting block_size
            for split_name, regions in split_regions.items():
                if not regions:
                    continue

                # Sort regions by start position
                regions.sort()

                # Group regions into blocks
                current_start, current_end = regions[0]
                for region_start, region_end in regions[1:]:
                    # If we've reached block size or it's the last region or not extending the current block, create a block
                    if (region_end - current_start >= block_size or
                        (region_start, region_end) == regions[-1] or
                        region_start != current_end):
                        indices[f"{split_name}_indices"].append((file_i, current_start, current_end))
                        current_start, current_end = region_start, region_end
                        # last region
                        if (region_start, region_end) == regions[-1]:
                            indices[f"{split_name}_indices"].append((file_i, region_start, region_end))
                    else:
                        current_end = region_end
        indices["train_n_items"] = n_items["train"]
        indices["val_n_items"] = n_items["val"]
        indices["test_n_items"] = n_items["test"]
        return indices

    @profile
    @staticmethod
    def collate_fn(batch):
        """Collate function for perturbation dataset."""
        if len(batch) == 0:
            return batch

        # Initialize collated batch
        collated = {}

        # Get keys from first sample
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch]



            if key in ['pert_cell_g', 'ctrl_cell_g', 'ctrl_cell_emb', 'pert_cell_emb']:
                # Convert scipy sparse matrices to torch sparse tensors and stack
                collated[key] = collate_sparse_matrices_torch_direct(values)

            elif key in ['pert_emb', 'cell_type_onehot', 'batch_onehot']:
                # Stack regular tensors
                collated[key] = torch.stack(values)

            elif key in ['pert_name', 'cell_type']:
                # Handle string arrays - concatenate
                collated[key] = np.concatenate(values)

            elif key in ['pert_cell_barcode', 'ctrl_cell_barcode']:
                # Handle barcode arrays - stack as numpy arrays
                collated[key] = np.stack(values) if values[0] is not None else None

            else:
                # Default: try to stack as tensors
                try:
                    collated[key] = torch.stack(values)
                except Exception:
                    # If stacking fails, keep as list
                    collated[key] = values

        return collated

    def setup(self, stage):
        """Setup datasets for different stages."""
        dataset_kwargs = {
            'pert_embedding_file': self.pert_embedding_file,
            'hvg_file': self.hvg_file,
            'cell_type_label': self.cell_type_label,
            'target_label': self.target_label,
            'control_label': self.control_label,
            'batch_label': self.batch_label,
            'use_batches': self.use_batches,
            'group_size_S': self.group_size_S,
            'barcodes': self.barcodes
        }

        if stage == "fit":
            self.train_ds = self.dataset.create_distributed_ds(
                self.indices, self.sparse_keys, "train", **dataset_kwargs
            )
            self.val_ds = self.dataset.create_distributed_ds(
                self.indices, self.sparse_keys, "val", **dataset_kwargs
            )
        if stage == "test":
            self.val_ds = self.dataset.create_distributed_ds(
                self.indices, self.sparse_keys, "test", **dataset_kwargs
            )
        if stage == "predict":
            self.predict_ds = self.dataset.create_distributed_ds(
                self.indices, self.sparse_keys, "train", **dataset_kwargs
            )