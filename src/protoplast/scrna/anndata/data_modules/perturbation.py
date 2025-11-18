import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import anndata
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
from typing import Optional, Sequence, Union, Dict, List, Tuple
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.distributed as td
import os
import scipy.sparse
from line_profiler import profile
from protoplast.scrna.anndata.data_modules.utils import make_onehot_encoding_map
from protoplast.scrna.anndata.torch_dataloader import DistributedAnnDataset, AnnDataModule

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)




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
        group_size_S: int = 32,
        barcodes: bool = False,
        n_items: int = None,
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
        self.n_items = n_items

        self.pert_embedding = torch.load(pert_embedding_file)

        # Initialize control region tracking per cell type
        self.cell_type_ctrl_regions = {}  # {cell_type: (start, end)} per file
        
        self._initialize_region_mappings()
        self._initialized_worker_info = False

    def _get_mat_by_range(self, adata: anndata.AnnData, start: int, end: int):
        """Get matrix by range."""
        if len(self.sparse_keys) == 1:
            mat = getattr(adata, self.sparse_keys[0])[start:end]
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
        
        batches_flattened = np.concatenate([[f"f{i}_"] * ad.n_obs + ad.obs[self.batch_label].tolist() for i, ad in enumerate(adatas)]).flatten()
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
                
                if cell_type not in self.cell_type_ctrl_regions:
                    self.cell_type_ctrl_regions[cell_type] = {}
                self.cell_type_ctrl_regions[cell_type][file_i] = (ctrl_start, ctrl_end)
        
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
    
    def get_celltype_onehot(self, cell_type: str):
        """Get onehot encoding for cell type."""
        return self.cell_types_onehot_map[cell_type]
    
    def get_batch_onehot(self, batch: str):
        """Get onehot encoding for batch."""
        return self.batches_onehot_map[batch]
    
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
        adata = self.adatas[file_i]
        X = self._get_mat_by_range(adata, start, end)
        
        # Get targets in this region
        targets = adata.obs[self.target_label][start:end].values
        target_counts = pd.Series(targets).value_counts()
        target_cumsum = target_counts.cumsum()
        target_cumsum = np.insert(target_cumsum.values, 0, 0)
        
        for i in range(1, len(target_counts) + 1):
            start_i, end_i = target_cumsum[i-1], target_cumsum[i]
            target = target_counts.index[i-1]
            
            X_pert = X[start_i:end_i]
            original_cell_indices = np.arange(start + start_i, start + end_i)

            for j in range(0, X_pert.shape[0], self.group_size_S):
                X_pert_group = X_pert[j:j+self.group_size_S]
                cell_indices_group = original_cell_indices[j:j+self.group_size_S]

                if X_pert_group.shape[0] < self.group_size_S:
                    # Sample with replacement
                    sample_indices = np.random.choice(X_pert_group.shape[0], self.group_size_S, replace=True)
                    X_pert_group = X_pert_group[sample_indices]
                    cell_indices_group = cell_indices_group[sample_indices]

                yield X_pert_group, cell_indices_group, target
    
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
        
        if (end - start) > target_number:
            start_pos = np.random.randint(0, end - start - target_number + 1)
            end_pos = start_pos + target_number
            X = self._get_mat_by_range(self.adatas[file_i], start_pos, end_pos)
            barcodes = self.adatas[file_i].obs_names[start_pos:end_pos].values
        else:
            X = self._get_mat_by_range(self.adatas[file_i], start, end)
            barcodes = self.adatas[file_i].obs_names[start:end].values

        # if not enough cells, sample with replacement
        if X.shape[0] < target_number:
            indices = np.random.choice(X.shape[0], target_number, replace=True)
            X = X[indices]
            barcodes = barcodes[indices]

        return X, barcodes

    def __len__(self):
        return self.n_items

    def __iter__(self):
        """Iterate over perturbation samples."""
        if not self._initialized_worker_info:
            self._init_worker_info()
        # Load anndata objects in backed mode
        if self.adatas is None:
            self.adatas = [anndata.read_h5ad(f, backed="r") for f in self.files]
            
        for region_idx, region in enumerate(self.batches):
            file_i, start, end = region

            # Worker/rank filtering
            if not (region_idx % self.ray_size == self.ray_rank and region_idx % self.nworkers == self.wid):
                continue

            try:
                # Get cell type (each file is one cell type only)
                cell_type = self.adatas[file_i].obs[self.cell_type_label].iloc[0]
                
                # Process each target in this region
                for X_pert, cell_indices, target in self._produce_data(file_i, start, end):
                    
                    # Get batches for perturbation cells
                    batches = self.adatas[file_i].obs[self.batch_label].iloc[cell_indices].values
                    
                    # Get control cells with matching covariates
                    X_ctrl, ctrl_barcodes = self.sampling_control(cell_type, file_i, self.group_size_S)
                    
                    # Get embeddings and onehot encodings
                    pert_emb = self._get_pert_embedding(target)
                    cell_type_onehot = self.get_celltype_onehot(cell_type)
                    batch_onehots = [self.get_batch_onehot(batch) for batch in batches]
                    
                    # Get barcodes for perturbation cells if needed
                    pert_barcodes = None
                    if self.barcodes:
                        pert_barcodes = self.adatas[file_i].obs_names[cell_indices].values
                    
                    # Create sample dictionary
                    sample = {
                        "pert_cell_emb": X_pert, # scipy csr matrix [S, G]
                        "ctrl_cell_emb": X_ctrl, # scipy csr matrix [S, G]
                        "pert_emb": pert_emb, # tensor [5102]
                        "pert_name": np.array([target]), 
                        "cell_type": np.array([cell_type]), # str
                        "cell_type_onehot": torch.stack([cell_type_onehot] * self.group_size_S), # tensor [S, n_cell_type]
                        "batch_onehot": torch.stack(batch_onehots), # tensor [S, n_batches]
                    }
                    
                    # Add barcodes if enabled
                    if self.barcodes:
                        sample["pert_cell_barcode"] = pert_barcodes # np.ndarray [S]
                        sample["ctrl_cell_barcode"] = ctrl_barcodes # np.ndarray [S]
                        
                    yield sample
                
            except Exception as e:
                logger.warning(f"Error processing region {region}: {e}")
                

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
        block_size: int = 2048,
        **kwargs
    ):
        if not kwargs.get("sparse_keys", None):
            kwargs["sparse_keys"] = ["X"]

        indices = self.build_indices(files, target_label, control_label, block_size)
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

    @staticmethod
    def build_indices(files: list[str], 
                      target_label: str = "target_gene", 
                      control_label: str = "non-targeting", 
                      block_size: int = 2048,
                      group_size_S: int = 32):
        """Build indices for perturbation dataset."""
        indices = {
            "files": files,
            "train_indices": [],
            "val_indices": [],
            "test_indices": [],
            "metadata": {},
        }
        n_items = 0
        # more split logic will go here, but only train for now
        for file_i, file in enumerate(files):
            adata = anndata.read_h5ad(file, backed="r")
            
            # Create regions that respect target boundaries and minimum block size
            target_counts = adata.obs.groupby(target_label, observed=True).agg({target_label: "count"})
            # each target will be yieled at least one item, but each target has more than group_size_S items, n items = n cells // group_size_S
            non_control_targets = target_counts[target_counts.index != control_label]
            items = np.array(non_control_targets[target_label]) // group_size_S
            items[items == 0] = 1
            n_items += np.sum(items)

            cumsum = np.cumsum(target_counts[target_label])
            cumsum = np.insert(cumsum, 0, 0)
            
            # Group consecutive targets into blocks of minimum size
            current_start = 0
            for i in range(1, len(target_counts) + 1):
                target_end = cumsum[i]
                current_target = target_counts.index[i-1]

                if current_target == control_label:
                    # check if we have a lagging region , then close it
                    if current_start != indices["train_indices"][-1][2]:
                        indices["train_indices"].append((file_i, current_start, cumsum[i-1]))
                    # start new region
                    current_start = target_end
                    continue

                # If we've reached block size or it's the last target, create a region
                if target_end - current_start >= block_size or i == len(target_counts):
                    indices["train_indices"].append((file_i, current_start, target_end))
                    current_start = target_end
        indices["train_n_items"] = n_items
        indices["val_n_items"] = 0  
        indices["test_n_items"] = 0
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
            
            if key in ['pert_cell_emb', 'ctrl_cell_emb']:
                # Convert scipy sparse matrices to torch sparse tensors and stack
                torch_sparse_tensors = []
                for val in values:
                    if scipy.sparse.issparse(val):
                        # Convert to COO format first, as we can't stack csr tensors. "Sparse CSR tensors do not have is_contiguous"
                        coo = val.tocoo()
                        # Create torch sparse tensor
                        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
                        values_tensor = torch.from_numpy(coo.data).float()
                        sparse_tensor = torch.sparse_coo_tensor(indices, values_tensor, coo.shape)
                        torch_sparse_tensors.append(sparse_tensor)
                    else:
                        # If already tensor, just append
                        torch_sparse_tensors.append(val)
                
                # Stack sparse tensors
                collated[key] = torch.stack(torch_sparse_tensors)
                
            elif key in ['pert_emb', 'cell_type_onehot', 'batch_onehot']:
                # Stack regular tensors
                collated[key] = torch.stack(values)
                
            elif key == 'pert_name':
                # Handle string arrays - concatenate
                collated[key] = np.concatenate(values)
                
            elif key in ['pert_cell_barcode', 'ctrl_cell_barcode']:
                # Handle barcode arrays - stack as numpy arrays
                collated[key] = np.stack(values) if values[0] is not None else None
                
            else:
                # Default: try to stack as tensors
                try:
                    collated[key] = torch.stack(values)
                except:
                    # If stacking fails, keep as list
                    collated[key] = values
        
        return collated
    
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

if __name__ == "__main__":
    import glob
    files = glob.glob("/mnt/hdd2/tan/competition_support_set_sorted/*.h5")
    dm = PerturbationDataModule(
        files=files,
        barcodes=True,
        pert_embedding_file="/mnt/hdd2/tan/competition_support_set/ESM2_pert_features.pt"
    )
    dm.setup(stage="fit")
    dataloader = DataLoader(dm.train_ds, batch_size=16, collate_fn=PerturbationDataModule.collate_fn, num_workers=8, pin_memory=True, persistent_workers=False)
    iter = 0
    for batch in dm.train_ds:
        print("Batch keys:", batch.keys())
        print("pert_cell_emb shape:", batch['pert_cell_emb'].shape)
        print(batch['pert_cell_emb'])
        print("ctrl_cell_emb shape:", batch['ctrl_cell_emb'].shape)
        print(batch['ctrl_cell_emb'])
        print("cell_type_onehot shape:", batch['cell_type_onehot'].shape)
        print("batch_onehot shape:", batch['batch_onehot'].shape)
        print("pert_emb shape:", batch['pert_emb'].shape)
        print("pert_name:", batch['pert_name'])
        print("pert_cell_barcode shape:", batch['pert_cell_barcode'].shape)
        print(batch['pert_cell_barcode'])
        print("ctrl_cell_barcode shape:", batch['ctrl_cell_barcode'].shape)
        print(batch['ctrl_cell_barcode'])
        print("cell_type:", batch['cell_type'])
        break