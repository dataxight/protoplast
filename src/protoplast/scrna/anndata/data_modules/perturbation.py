import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import scplode as sp
import anndata
import numpy as np
from collections import defaultdict
import logging
from typing import Optional, Sequence, Union, Dict, List, Tuple
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import scipy.sparse

from protoplast.scrna.anndata.data_modules.utils import make_onehot_encoding_map
from protoplast.scrna.anndata.torch_dataloader import DistributedAnnDataset, AnnDataModule

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def expand_target_regions_to_blocks(target_regions: List[Tuple], block_size: int) -> List[Tuple]:
    """
    Expand target_regions into smaller consecutive regions with max size block_size.
    
    Args:
        target_regions: List of (file_i, target_i, start_i, end_i) tuples
        block_size: Maximum block size
        
    Returns:
        List of expanded regions with max size block_size
    """
    expanded_regions = []
    
    for file_i, target_i, start_i, end_i in target_regions:
        region_size = end_i - start_i
        
        if region_size <= block_size:
            # Region is smaller than or equal to block size, keep as is
            expanded_regions.append((file_i, target_i, start_i, end_i))
        else:
            # Split region into blocks of size block_size
            current_start = start_i
            while current_start < end_i:
                current_end = min(current_start + block_size, end_i)
                expanded_regions.append((file_i, target_i, current_start, current_end))
                current_start = current_end
                
    return expanded_regions


class PerturbationDataset(DistributedAnnDataset):
    """
    PyTorch Dataset for perturbation scRNA-seq stored in AnnData h5ad files.
    
    The h5ad files are pre-sorted by target-gene then batch, each file contains 
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
        cell_type_label: str = "cell_type",
        target_label: str = "target_gene", 
        control_label: str = "non-targeting",
        batch_label: str = "batch_var", 
        use_batches: bool = True,
        group_size_S: int = 30,
        barcodes: bool = False,
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
        self.group_size_S = group_size_S
        self.barcodes = barcodes

        self.pert_embedding = torch.load(pert_embedding_file)

        # Initialize region tracking dictionaries
        self.target_gene_regions = defaultdict(list)  # {target_gene: [(file_i, start, end), ...]}
        self.cell_type_regions = defaultdict(list)    # {cell_type: [(file_i, start, end), ...]}
        self.cell_type_ctrl_regions = defaultdict(list)  # {cell_type: [(start, end), ...]} per file
        
        self._initialize_region_mappings()

    def _get_mat_by_range(self, adata: anndata.AnnData, start: int, end: int):
        """Get matrix by range."""
        if len(self.sparse_keys) == 1:
            mat = getattr(adata, self.sparse_keys[0])[start:end]
            if not scipy.sparse.issparse(mat):
                mat = scipy.sparse.csr_matrix(mat)
            return mat
        else:
            raise ValueError(f"Multiple sparse keys are not supported for perturbation dataset: {self.sparse_keys}")
    
    def _initialize_region_mappings(self):
        """Initialize region mappings for target genes, cell types, and controls."""
        # Load adatas to analyze structure
        adatas = [anndata.read_h5ad(f, backed="r") for f in self.files]
        
        logger.info(f"n_cells: {np.array([ad.n_obs for ad in adatas]).sum()}")
        logger.info(f"n_genes: {adatas[0].n_vars}")
        
        # Build onehot mappings
        cell_types_flattened = np.concatenate([ad.obs[self.cell_type_label].tolist() for ad in adatas]).flatten()
        self.cell_types_onehot_map = make_onehot_encoding_map(np.unique(cell_types_flattened))
        logger.info(f"Total unique cell types: {len(self.cell_types_onehot_map)}")
        
        batches_flattened = np.concatenate([[f"f{i}_"] * ad.n_obs + ad.obs[self.batch_label].tolist() for i, ad in enumerate(adatas)]).flatten()
        self.batches_onehot_map = make_onehot_encoding_map(np.unique(batches_flattened))
        logger.info(f"Total unique batches: {len(self.batches_onehot_map)}")
        
        # Build region mappings from batches (assuming indices structure)
        for region in self.batches:  # batches contains the region information
            file_i, target, start, end = region
            
            # Get cell type for this file (each file is one cell type only)
            cell_type = adatas[file_i].obs[self.cell_type_label].iloc[0]
            
            # Track target gene regions
            self.target_gene_regions[target].append((file_i, start, end))
            
            # Track cell type regions
            self.cell_type_regions[cell_type].append((file_i, start, end))
            
            # Track control regions for each cell type
            if target == self.control_label:
                if len(self.cell_type_ctrl_regions[cell_type]) <= file_i:
                    self.cell_type_ctrl_regions[cell_type].extend([(0, 0)] * (file_i + 1 - len(self.cell_type_ctrl_regions[cell_type])))
                
                # Update or set the control region for this file
                current_start, current_end = self.cell_type_ctrl_regions[cell_type][file_i]
                if current_start == 0 and current_end == 0:
                    self.cell_type_ctrl_regions[cell_type][file_i] = (start, end)
                else:
                    # Extend the region if needed
                    new_start = min(current_start, start)
                    new_end = max(current_end, end)
                    self.cell_type_ctrl_regions[cell_type][file_i] = (new_start, new_end)
        
        # Get all cell barcodes if needed
        if self.barcodes:
            self.cell_barcodes_flattened = np.concatenate([ad.obs_names.tolist() for ad in adatas]).flatten()

    @classmethod
    def create_distributed_ds(cls, indices: dict, sparse_keys: list[str], mode: str = "train", **kwargs):
        """
        Create distributed dataset with expanded target regions.
        
        indices format:
        {
            "files": [list of file paths],
            "train_indices": [...target_regions],  # List of (file_i, target_i, start_i, end_i)
            "metadata": {},
            "sparse_keys": ["X"]
        }
        """
        # Get block size from kwargs, default to 64
        block_size = kwargs.pop('block_size', 64)
        
        # Expand target regions into blocks
        target_regions = indices[f"{mode}_indices"]
        expanded_regions = expand_target_regions_to_blocks(target_regions, block_size)
        
        # Update indices with expanded regions
        updated_indices = indices.copy()
        updated_indices[f"{mode}_indices"] = expanded_regions
        
        return cls(file_paths=indices["files"], 
                    indices=updated_indices[f"{mode}_indices"],
                    metadata=updated_indices["metadata"],
                    sparse_keys=sparse_keys, **kwargs)

    def _get_pert_embedding(self, pert_id: str):
        """Get perturbation embedding, create zero embedding if not found."""
        if pert_id not in self.pert_embedding:
            # create all zero embedding
            self.pert_embedding[pert_id] = torch.zeros(next(iter(self.pert_embedding.values())).shape[0])
        return self.pert_embedding[pert_id]
    
    def get_celltype_onehot(self, cell_type: str):
        """Get onehot encoding for cell type."""
        return self.cell_types_onehot_map[cell_type]
    
    def get_batch_onehot(self, batch: str):
        """Get onehot encoding for batch."""
        return self.batches_onehot_map[batch]
    
    
    def _sample_cells(self, X: scipy.sparse.csr_matrix, target_number: int, barcodes: np.ndarray = None):
        """
        Helper function to sample cells to target number.
        
        Args:
            X: Input sparse matrix (csr)
            target_number: Target number of cells
            barcodes: Optional barcodes array
            
        Returns:
            Tuple of (sampled_X, sampled_barcodes)
        """
        if X.shape[0] < target_number:
            # Sampling with replacement to pad
            indices = np.random.choice(X.shape[0], target_number, replace=True)
            X_sampled = X[indices]
            barcodes_sampled = barcodes[indices] if barcodes is not None else None
                
        elif X.shape[0] > target_number:
            # Random slicing
            start_pos = np.random.randint(0, X.shape[0] - target_number + 1)
            end_pos = start_pos + target_number
            X_sampled = X[start_pos:end_pos]
            barcodes_sampled = barcodes[start_pos:end_pos] if barcodes is not None else None
        else:
            # Perfect match
            X_sampled = X
            barcodes_sampled = barcodes if barcodes is not None else None
                
        return X_sampled, barcodes_sampled
    
    def sampling_control(self, cell_type: str, batch: str, file_i: int, target_number: int):
        """
        Sample control cells matching covariate cell type and batch.
        
        Args:
            cell_type: Cell type to match
            batch: Batch to match 
            file_i: File index
            target_number: Number of cells to sample
            
        Returns:
            Sampled control cell matrix and barcodes (if enabled)
        """
        if cell_type not in self.cell_type_ctrl_regions:
            raise ValueError(f"No control regions found for cell type: {cell_type}")
            
        region_of_files = self.cell_type_ctrl_regions[cell_type]
        if file_i >= len(region_of_files):
            raise ValueError(f"File index {file_i} not found in control regions for cell type {cell_type}")
            
        start, end = region_of_files[file_i]
        if start == 0 and end == 0:
            raise ValueError(f"No control region found for file {file_i} and cell type {cell_type}")
            
        # Get batches in the control region
        batches = self.adatas[file_i].obs[self.batch_label][start:end].values
        batch_mask = batches == batch
        batch_indices = np.where(batch_mask)[0]
        
        if len(batch_indices) == 0:
            # If no exact batch match, use all control cells from this cell type
            batch_indices = np.arange(end - start)
            
        # Adjust indices to global range
        global_indices = batch_indices + start
        b_start, b_end = global_indices[0], global_indices[-1] + 1
        
        X = self._get_mat_by_range(self.adatas[file_i], b_start, b_end)
        
        # Get barcodes if needed
        barcodes = None
        if self.barcodes:
            barcodes = self.adatas[file_i].obs_names[b_start:b_end].values
            
        # Sample cells using helper function
        X_sampled, ctrl_barcodes = self._sample_cells(X, target_number, barcodes)
                
        return X_sampled, ctrl_barcodes

    def transform(self, file_i: int, target: str, start: int, end: int):
        """
        Transform a region of cells from perturbation data.
        
        Args:
            file_i: File index
            target: Target gene
            start: Start index
            end: End index
            
        Returns:
            Tuple of (X_pert, batch, pert_barcodes)
        """
        # Get batches in this region
        batches = self.adatas[file_i].obs[self.batch_label][start:end].values
        
        # Find most represented batch
        unique_batches, counts = np.unique(batches, return_counts=True)
        most_common_batch = unique_batches[np.argmax(counts)]
        
        # Get cells from the most common batch
        batch_mask = batches == most_common_batch
        batch_indices = np.where(batch_mask)[0]
        
        # Adjust indices to global range
        global_indices = batch_indices + start
        b_start, b_end = global_indices[0], global_indices[-1] + 1
        
        X = self._get_mat_by_range(self.adatas[file_i], b_start, b_end)
        
        # Get barcodes if needed
        barcodes = None
        if self.barcodes:
            barcodes = self.adatas[file_i].obs_names[b_start:b_end].values
            
        # Sample cells using helper function
        X_sampled, pert_barcodes = self._sample_cells(X, self.group_size_S, barcodes)
                
        return X_sampled, most_common_batch, pert_barcodes

    def __iter__(self):
        """Iterate over perturbation samples."""
        gidx = 0
        
        # Load anndata objects in backed mode
        if self.adatas is None:
            self.adatas = [anndata.read_h5ad(f, backed="r") for f in self.files]
            
        for region in self.batches:
            file_id, target, start, end = region
            
            # Skip control samples and worker/rank filtering
            if target == self.control_label or not (gidx % self.ray_size == self.ray_rank and gidx % self.nworkers == self.wid):
                gidx += 1
                continue
                
            try:
                # Get perturbation cells
                X_pert, batch, pert_barcodes = self.transform(file_id, target, start, end)
                
                # Get cell type (each file is one cell type only)
                cell_type = self.adatas[file_id].obs[self.cell_type_label].iloc[0]
                
                # Get control cells with matching covariates
                X_ctrl, ctrl_barcodes = self.sampling_control(cell_type, batch, file_id, self.group_size_S)
                
                # Get embeddings and onehot encodings
                pert_emb = self._get_pert_embedding(target)
                cell_type_onehot = self.get_celltype_onehot(cell_type)
                batch_onehot = self.get_batch_onehot(batch)
                
                # Create sample dictionary
                sample = {
                    "pert_cell_emb": X_pert,
                    "ctrl_cell_emb": X_ctrl,
                    "pert_emb": pert_emb,
                    "pert_name": target,
                    "cell_type_onehot": cell_type_onehot,
                    "batch_onehot": batch_onehot,
                }
                
                # Add barcodes if enabled
                if self.barcodes:
                    sample["pert_cell_barcode"] = pert_barcodes
                    sample["ctrl_cell_barcode"] = ctrl_barcodes
                    
                yield sample
                
            except Exception as e:
                logger.warning(f"Error processing region {region}: {e}")
                
            gidx += 1


class PerturbationDataModule(AnnDataModule):
    """
    PyTorch Lightning DataModule for perturbation scRNA-seq data.
    
    Subclasses AnnDataModule to work with PerturbationDataset.
    """
    def __init__(
        self,
        files: list[str],
        pert_embedding_file: str,
        num_workers: int = None,
        prefetch_factor: int = 2,
        cell_type_label: str = "cell_type",
        target_label: str = "target_gene",
        control_label: str = "non-targeting", 
        batch_label: str = "batch_var",
        use_batches: bool = True,
        group_size_S: int = 32,
        barcodes: bool = False,
        block_size: int = 64,
        **kwargs
    ):
        if not kwargs.get("sparse_keys", None):
            kwargs["sparse_keys"] = ["X"]

        indices = self.build_indices(files)
        # Initialize parent with PerturbationDataset
        super().__init__(
            indices=indices,
            dataset=PerturbationDataset,
            prefetch_factor=prefetch_factor,
            **kwargs
        )

        if not num_workers is None:
            self.loader_config["num_workers"] = num_workers
        
        # Store perturbation-specific parameters
        self.pert_embedding_file = pert_embedding_file
        self.cell_type_label = cell_type_label
        self.target_label = target_label
        self.control_label = control_label
        self.batch_label = batch_label
        self.use_batches = use_batches
        self.group_size_S = group_size_S
        self.barcodes = barcodes
        self.block_size = block_size

    @staticmethod
    def build_indices(files: list[str], target_label: str = "target_gene"):
        """Build indices for perturbation dataset."""
        indices = {
            "files": files,
            "train_indices": [],
            "val_indices": [],
            "test_indices": [],
            "metadata": {},
        }
        # more split logic will go here, but only train for now
        for file_i, file in enumerate(files):
            # we count the number of cells per each target gene then build the index tuple (file_i, target_i, start_i, end_i)
            adata = anndata.read_h5ad(file, backed="r")
            target_counts = adata.obs.groupby(target_label, observed=True).agg({target_label: "count"})
            cumsum = np.cumsum(target_counts[target_label])
            # prepend 0
            cumsum = np.insert(cumsum, 0, 0)
            for i in range(1, len(target_counts) + 1):
                start, end = cumsum[i-1], cumsum[i]
                target = target_counts.index[i-1]
                indices["train_indices"].append((file_i, target, start, end))
        return indices

    def setup(self, stage):
        """Setup datasets for different stages."""
        dataset_kwargs = {
            'pert_embedding_file': self.pert_embedding_file,
            'cell_type_label': self.cell_type_label,
            'target_label': self.target_label,
            'control_label': self.control_label,
            'batch_label': self.batch_label,
            'use_batches': self.use_batches,
            'group_size_S': self.group_size_S,
            'barcodes': self.barcodes,
            'block_size': self.block_size
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

if __name__ == "__main__":
    dm = PerturbationDataModule(
        files=["/mnt/hdd2/tan/competition_support_set_sorted/jurkat.h5"],
        pert_embedding_file="/mnt/hdd2/tan/competition_support_set/ESM2_pert_features.pt"
    )
    dm.setup(stage="fit")
    print(next(iter(dm.train_ds)))