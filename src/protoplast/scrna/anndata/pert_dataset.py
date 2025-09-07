import torch
from torch.utils.data import Dataset
import scplode as sp
import anndata as ad
import numpy as np
from collections import defaultdict
import logging

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