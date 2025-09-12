
import os
import random
import warnings
from collections.abc import Callable

import lightning.pytorch as pl
import numpy as np
import scipy.sparse as sp
import torch
import torch.distributed as td
from torch.utils.data import DataLoader, get_worker_info
import anndata
import scipy as scp 

import anndata
from protoplast.patches.anndata_read_h5ad_backed import apply_read_h5ad_backed_patch
from protoplast.patches.anndata_remote import apply_file_backing_patch

apply_file_backing_patch()
apply_read_h5ad_backed_patch()


def ann_split_data(
    file_paths: list[str],
    batch_size: int,
    test_size: float | None = None,
    validation_size: float | None = None,
    random_seed: int | None = 42,
    metadata_cb: Callable[[anndata.AnnData, dict], None] | None = None,
    is_shuffled: bool = True,
):
    def to_batches(n):
        return [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    rng = random.Random(random_seed) if random_seed else random.Random()

    # First pass: compute total batches across all files
    file_batches = []
    total_batches = 0
    metadata = dict()
    for i, fp in enumerate(file_paths):
        ad = anndata.read_h5ad(fp, backed="r")
        if i == 0 and metadata_cb:
            metadata_cb(ad, metadata)

        n_obs = ad.n_obs
        if batch_size > n_obs:
            warnings.warn(
                f"Batch size ({batch_size}) is greater than number of observations "
                f"in file {fp} ({n_obs}). Only one batch will be created.",
                stacklevel=2,
            )

        batches = to_batches(n_obs)
        total_batches += len(batches)
        file_batches.append(batches)

    # Safety check
    if (test_size or 0) + (validation_size or 0) > 1:
        raise ValueError("test_size + validation_size must be <= 1")

    # How many batches should go to validation & test globally?
    val_total = int(total_batches * validation_size) if validation_size else 0
    test_total = int(total_batches * test_size) if test_size else 0

    train_datas, validation_datas, test_datas = [], [], []

    # Second pass: allocate splits proportionally per file
    for batches in file_batches:
        if is_shuffled:
            rng.shuffle(batches)
        n = len(batches)

        val_n = int(round(n / total_batches * val_total)) if validation_size else 0
        test_n = int(round(n / total_batches * test_total)) if test_size else 0

        val_split = batches[:val_n]
        test_split = batches[val_n : val_n + test_n]
        train_split = batches[val_n + test_n :]

        validation_datas.append(val_split)
        test_datas.append(test_split)
        train_datas.append(train_split)

    return dict(
        files=file_paths,
        train_indices=train_datas,
        val_indices=validation_datas,
        test_indices=test_datas,
        metadata=metadata,
    )


def cell_line_metadata_cb(ad: anndata.AnnData, metadata: dict):
    """
    Example callback for adding cell line metadata when you use this
    with DistributedAnnDataset it will automatically assign all of the key
    and values to an instance variable
    for example metadata["cell_lines"] will be avaliable as self.cell_lines
    in DistributedAnnDataset where you can use it for transformation
    """
    metadata["cell_lines"] = ad.obs["cell_line"].cat.categories.to_list()
    metadata["num_genes"] = ad.var.shape[0]
    metadata["num_classes"] = len(metadata["cell_lines"])


class DistributedAnnDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_paths: list[str],
        indices: list[list[int]],
        metadata: dict,
        sparse_keys: list[str],
    ):
        # use first file as reference first
        self.files = file_paths
        self.sparse_keys = sparse_keys
        # map each gene to an index
        for k, v in metadata.items():
            setattr(self, k, v)
        self.metadata = metadata
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
        self.batches = indices

    @classmethod
    def create_distributed_ds(cls, indices: dict, sparse_keys: list[str], mode: str = "train"):
        """
        indices is in the following format
        {
            "files": [path to anndata must correspond to indices],
            "train_indices": [[correspond to files[0]], [correspond to files[i]] ],
            "test_indices": [[correspond to files[0]], [correspond to files[i]] ],
            "metadata": {
                ...,
                depends on metadata_cb read more on cell_line_metadata_cb
            }
        }
        """
        return cls(indices["files"], indices[f"{mode}_indices"], indices["metadata"], sparse_keys)

    def _process_sparse(self, mat) -> torch.Tensor:
        if sp.issparse(mat):
            return torch.sparse_csr_tensor(
                torch.from_numpy(mat.indptr).long(),
                torch.from_numpy(mat.indices).long(),
                torch.from_numpy(mat.data).float(),
                mat.shape,
            )
        return torch.from_numpy(mat).float()

    def transform(self, start: int, end: int):
        mats = []
        for k in self.sparse_keys:
            if "." in k:
                attr, attr_k = k.split(".")
                mat = getattr(self.ad, attr)[attr_k][start:end]
                mats.append(self._process_sparse(mat))
            else:
                mat = getattr(self.ad, k)[start:end]
                mats.append(self._process_sparse(mat))
        if len(mats) == 1:
            return mats[0]
        if mats[0].shape[0] == 0:
            return None
        return tuple(mats)

    def __iter__(self):
        gidx = 0
        for fidx, f in enumerate(self.files):
            self.ad = anndata.read_h5ad(f, backed="r")
            for start, end in self.batches[fidx]:
                if gidx % self.ray_size == self.ray_rank and gidx % self.nworkers == self.wid:
                    data = self.transform(start, end)
                    if data is not None:
                        yield self.transform(start, end)
                gidx += 1


class BlockBasedAnnDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_paths: list[str],
        ds_batch_size: int,
        block_size: int,
        load_factor: int,
        sparse_keys: list[str],
        random_seed: int | None = 42,
        mode: str = "train"
    ):
        self.files = file_paths
        self.adatas = []
        self.batch_size = ds_batch_size # use ds_ just to not collide with the batch_size argument for loader
        self.block_size = block_size
        self.sparse_keys = sparse_keys
        self.random_seed = random_seed
        self.mode = mode
        if not isinstance(load_factor, int):
            raise ValueError("load_factor must be an integer")
        self.block_group_size = load_factor 

        if int(self.block_size * load_factor) % int(self.batch_size) != 0:
            raise ValueError("block_size * load_factor must be divisible by batch_size")
        
        # Initialize worker and distributed training info
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
            
        # Load anndata objects and create block ranges
        self.block_ranges = []
        self.n_cells = 0
        self.n_obs = []
        
        for i, file_path in enumerate(self.files):
            ad = anndata.read_h5ad(file_path, backed="r")
            self.n_cells += ad.n_obs
            self.n_obs.append(ad.n_obs)
            
            # Create block ranges for this file
            n_cells = ad.n_obs
            file_ranges = []
            
            for start in range(0, n_cells, self.block_size):
                end = min(start + self.block_size, n_cells)
                file_ranges.append((i, start, end))
                
            self.block_ranges.extend(file_ranges)
        
        # Shuffle the block ranges
        if self.random_seed is not None:
            rng = random.Random(self.random_seed)
            rng.shuffle(self.block_ranges)
        else:
            random.shuffle(self.block_ranges)
            
    
    @classmethod
    def create_distributed_ds(
        cls, 
        file_paths: list[str], 
        ds_batch_size: int, 
        block_size: int, 
        load_factor: int,
        sparse_keys: list[str], 
        random_seed: int | None = 42,
        mode: str = "train"
    ):
        """
        Create a BlockBasedAnnDataset instance.
        
        Args:
            file_paths: List of paths to h5ad files
            batch_size: Size of batches to yield
            block_size: Size of blocks to read at once
            load_factor: Factor by which to load blocks
            sparse_keys: Keys for sparse matrices to extract
            metadata: Metadata dictionary to set as instance attributes
            random_seed: Random seed for shuffling block ranges
        """
        return cls(file_paths, ds_batch_size, block_size, load_factor, sparse_keys, random_seed, mode)
    
    def _process_sparse(self, mat) -> torch.Tensor:
        """Convert sparse matrix to torch tensor."""
        if sp.issparse(mat):
            return torch.sparse_csr_tensor(
                torch.from_numpy(mat.indptr).long(),
                torch.from_numpy(mat.indices).long(),
                torch.from_numpy(mat.data).float(),
                mat.shape,
            )
        return torch.from_numpy(mat).float()


    def _is_sparse(self, mat):
        return isinstance(mat, anndata._core.sparse_dataset._CSRDataset) or sp.issparse(mat)

    def _process_sparse2(self, mat, start, end):
        if self._is_sparse(mat):
            sampling_indices = range(start, end)

            # List of pointers to indicate number of non-zero values for each row
            # in a sparse matrix
            indptr = torch.zeros(len(sampling_indices) + 1).long()
        
            if isinstance(mat, scp.sparse._csr.csr_matrix):
                mat_indptr = mat.indptr
                mat_indices = mat.indices
                mat_data = mat.data        
            else:
                mat_indptr = mat._indptr
                mat_indices = mat._indices
                mat_data = mat._data
            
            # First pass compute indptr of the rows for pin-point non-zero columns in that row
            total_non_zeros = 0
            for i, row_num in enumerate(sampling_indices):
                # End index of the current row in indptr
                indptr[i + 1] = indptr[i] + (mat_indptr[row_num + 1] - mat_indptr[row_num])
            
                total_non_zeros += (indptr[i + 1] - indptr[i])
            
            # List of indices of non-zero columns in the rows and the data of those columns
            indices = torch.zeros(total_non_zeros).long()
            data = torch.zeros(total_non_zeros).float()
            shape = (len(sampling_indices), mat.shape[1])

            for i, row_num in enumerate(sampling_indices):
                # Column indices of non-zero val within the current row
                indices[indptr[i]:indptr[i+1]] = torch.from_numpy(mat_indices[mat_indptr[row_num]:mat_indptr[row_num+1]])
                
                # Data of non-zero
                data[indptr[i]:indptr[i+1]] = torch.from_numpy(mat_data[mat_indptr[row_num]:mat_indptr[row_num+1]])
            
            sparse_mat = sp.csr_matrix((data, indices, indptr), shape = shape)
            return sparse_mat
        
        return torch.from_numpy(mat[start:end, :]).float()
    
    def _get_mat_by_range(self, file_idx: int, start: int, end: int, sparse_key: str | None = None):
        """Helper function to get random items from a file
        this is in case we need to create padding data
        """
        mat = None
        
        adata = self.adatas[file_idx]
        if "." in sparse_key:
            attr, attr_k = sparse_key.split(".")
            mat = self._process_sparse2(getattr(adata, attr)[attr_k], start, end)
        else:
            mat = self._process_sparse2(getattr(adata, sparse_key), start, end)
        # just in case it is a dense matrix, convert it to a sparse matrix
        mat = sp.csr_matrix(mat)
        return mat

    def transform(self, X: torch.Tensor, cell_idx: np.ndarray):
        """
        X: torch.Tensor
        cell_idx: np.ndarray each item is a tuple of (file_idx, cell_idx)
        """
        return X, cell_idx

    def __iter__(self):
        gidx = 0

        # Since PyTorch seems to adopt lazy way of initializing workers, 
        # this means that the actual file opening has to happen inside of the__getitem__function of the Dataset wrapper. 
        # refer to https://stackoverflow.com/questions/46045512/h5py-hdf5-database-randomly-returning-nans-and-near-very-small-data-with-multi/52438133#52438133
        for file in self.files:
            self.adatas.append(anndata.read_h5ad(file, backed="r"))

        # Process blocks in groups of block_group_size
        for i in range(0, len(self.block_ranges), self.block_group_size):
            if gidx % self.ray_size == self.ray_rank and gidx % self.nworkers == self.wid:
                mats = []
                cell_idx = [] # each item is a tuple of (file_idx, cell_idx)
                
                # Collect block_group_size worth of data
                for j in range(self.block_group_size):
                    if i + j >= len(self.block_ranges):
                        break
                        
                    file_idx, start, end = self.block_ranges[i + j]
                    cell_idx += [(file_idx, cell_idx) for cell_idx in range(start, end)]
                    
                    # Get data for each sparse key and stack them
                    block_mats = []
                    for k in self.sparse_keys:
                        mat = self._get_mat_by_range(file_idx, start, end, k)
                        block_mats.append(mat)
                    
                    # If we have multiple sparse keys, we need to handle them appropriately
                    # For now, we'll assume we're working with the first sparse key for stacking
                    if len(block_mats) > 0:
                        mats.append(block_mats[0])  # Use first sparse key for main matrix
                
                if not mats:
                    gidx += 1
                    continue
                
                # Stack the matrices
                X = sp.vstack(mats)
                
                # Check if we need padding to make it divisible by batch_size
                # TODO: implement the drop_last logic here to skip the last batch if needed so we don't need to do padding
                if X.shape[0] % self.batch_size != 0:
                    remainder = X.shape[0] % self.batch_size
                    padding_needed = self.batch_size - remainder
                    
                    # Get random padding data
                    padding_mats = []
                    remaining_padding = padding_needed
                    
                    while remaining_padding > 0:
                        # Choose a random file
                        rfi = random.choice(range(len(self.files)))
                        n_obs = anndata.read_h5ad(self.files[rfi], backed="r").n_obs
                        
                        if n_obs < remaining_padding:
                            # If the file doesn't have enough cells, take all of them
                            chunk_size = n_obs
                            rstart = 0
                        else:
                            # Take a random chunk
                            chunk_size = min(remaining_padding, n_obs)
                            rstart = random.choice(range(n_obs - chunk_size + 1))
                        
                        # Get the padding data using the first sparse key
                        k = self.sparse_keys[0]
                        X_r = self._get_mat_by_range(rfi, rstart, rstart + chunk_size, k)

                        # concatenate cell_idx
                        cell_idx += [(rfi, cell_idx) for cell_idx in range(rstart, rstart + chunk_size)]

                        padding_mats.append(X_r)
                        remaining_padding -= chunk_size
                    
                    # Stack padding matrices
                    if padding_mats:
                        X_padding = sp.vstack(padding_mats)
                        # Concatenate original and padding
                        X = sp.vstack([X, X_padding])
                
                # Yield batches
                # shuffle X
                ridx = np.random.permutation(X.shape[0])
                X = X[ridx, :]
                # shuffle cell_idx too
                cell_idx = np.array(cell_idx)[ridx]
                n_batches = (self.block_size * self.block_group_size) // self.batch_size
                for bi in range(n_batches):
                    start_idx = bi * self.batch_size
                    end_idx = start_idx + self.batch_size
                    
                    batch_data = X[start_idx:end_idx, :]
                    yield self.transform(self._process_sparse(batch_data), cell_idx[start_idx:end_idx])
            
            gidx += 1


class DistributedCellLineAnnDataset(DistributedAnnDataset):
    """
    Example of how to extend DistributedAnnDataset to adapt it for cell line linear
    classification model here self.cell_lines is available through writing the
    metadata_cb correctly
    """

    def transform(self, start: int, end: int):
        X = super().transform(start, end)
        if X is None:
            return None
        line_ids = self.ad.obs["cell_line"].iloc[start:end]
        line_idx = np.searchsorted(self.cell_lines, line_ids)
        return X, torch.tensor(line_idx)


class AnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DistributedAnnDataset | BlockBasedAnnDataset,
        prefetch_factor: int,
        before_dense_cb: Callable[[torch.Tensor, str | int], torch.Tensor] = None,
        after_dense_cb: Callable[[torch.Tensor, str | int], torch.Tensor] = None,
        **dataset_kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
        self.loader_config = dict(batch_size=None, num_workers=num_threads, prefetch_factor=prefetch_factor)
        self.before_dense_cb = before_dense_cb
        self.after_dense_cb = after_dense_cb
        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage):
        # this is not necessary but it is here in case we want to download data to local node in the future
        if stage == "fit":
            self.train_ds = self.dataset.create_distributed_ds(**self.dataset_kwargs)
            self.val_ds = self.dataset.create_distributed_ds(**self.dataset_kwargs, mode="val")
        if stage == "test":
            self.val_ds = self.dataset.create_distributed_ds(**self.dataset_kwargs, mode="test")
        if stage == "predict":
            self.predict_ds = self.dataset.create_distributed_ds(**self.dataset_kwargs, mode="predict")

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.loader_config)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.loader_config)

    def test_dataloader(self):
        # for now not support testing for splitting will support it soon in the future
        return DataLoader(self.val_ds, **self.loader_config)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, **self.loader_config)

    def densify(self, x, idx: str | int = None):
        if isinstance(x, torch.Tensor):
            if self.before_dense_cb:
                x = self.before_dense_cb(x, idx)
            if x.is_sparse or x.is_sparse_csr:
                x = x.to_dense()
            if self.after_dense_cb:
                x = self.after_dense_cb(x, idx)
        return x

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if (type(batch) is list) or (type(batch) is tuple):
            return [self.densify(d, i) for i, d in enumerate(batch)]
        elif isinstance(batch, dict):
            return {k: self.densify(v, k) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return self.densify(batch)
        else:
            return batch