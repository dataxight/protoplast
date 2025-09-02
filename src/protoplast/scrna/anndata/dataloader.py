import torch
from torch.utils.data import Dataset
import scplode as sp
import anndata as ad
import numpy as np
from collections import defaultdict
import logging
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
    def __init__(self, h5ad_files, control_label="non-targeting", use_batches=True, device="cpu"):
        self.control_label = control_label
        self.use_batches = use_batches
        self.device = device
        self.h5ad_files = h5ad_files
        self.sp_adatas = []

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
        self.cell_types_flattened = np.concatenate([ad.obs["cell_type"].tolist() for ad in adatas]).flatten()
        # Map categorical labels to integer ids
        self.cell_types, self.y_index = np.unique(self.cell_types_flattened, return_inverse=True)
        logger.info(f"Total unique cell types: {len(self.cell_types)}")

        # get unique genes across all h5ad files
        self.perturb_vocab = np.concatenate([ad.var_names.tolist() for ad in adatas])
        self.perturb_flattened = [] 
        for adata in adatas:
            if "guide_id" in adata.obs:
                self.perturb_flattened = np.concatenate([self.perturb_flattened, adata.obs["guide_id"].tolist()])
            else:
                self.perturb_flattened = np.concatenate([self.perturb_flattened, adata.obs["gene"].tolist()])
        # Map categorical labels to integer ids
        self.perturbs_training, _ = np.unique(self.perturb_flattened, return_inverse=True)
        logger.info(f"Total unique perturbations in training set: {len(self.perturbs_training)}")
        self.perturbs_vocab, self.xp_index = np.unique([self.control_label] + self.perturb_vocab, return_inverse=True)

        logger.info(f"Total perturbations (vocab + training set): {len(self.perturbs_vocab)}")

        # get unique batches across all h5ad files
        self.batches_flattened = np.concatenate([[f"f{i}_"] * ad.n_obs + ad.obs["batch_var"].tolist() for i, ad in enumerate(adatas)]).flatten()
        self.batches, self.b_index = np.unique(self.batches_flattened, return_inverse=True)
        logger.info(f"Total unique batches: {len(self.batches)}")

        # Index controls by (y,b) for fast lookup
        self.control_index = np.where(self.perturb_flattened == self.control_label)[0]
        logger.info(f"Total control cells: {len(self.control_index)}")
        self.control_lookup = defaultdict(list)
        for i in self.control_index:
            self.control_lookup[(self.y_index[i], self.b_index[i])].append(i)

        # Store precomputed label tensors
        self.y = torch.tensor(self.y_index, dtype=torch.long, device=device)
        self.xp = torch.tensor(self.xp_index, dtype=torch.long, device=device)
        self.b = torch.tensor(self.b_index, dtype=torch.long, device=device)

    def __len__(self):
        return self.n_cells.sum()

    def _get_file_idx(self, idx):
        # Get the index of the file that contains the cell
        return np.where(idx < self.n_cells.cumsum())[0][0]

    def get_onehot_cell_types(self, idx):
        return torch.nn.functional.one_hot(self.y[idx], num_classes=len(self.cell_types))

    def get_onehot_perturbs(self, idx):
        return torch.nn.functional.one_hot(self.xp[idx], num_classes=len(self.perturbs_vocab))

    def get_onehot_batches(self, idx):
        return torch.nn.functional.one_hot(self.b[idx], num_classes=len(self.batches))

    def __getitem__(self, idx):
        # Fetch expression row, convert sparse → dense → torch
        file_idx = self._get_file_idx(idx)
        adata = self.sp_adatas[file_idx]
        barcode = self.cell_barcodes_flattened[idx]
        x = adata.get([barcode])
        x = torch.tensor(x, dtype=torch.float32, device=self.device)

        y_scalar = self.y_index[idx]
        logger.info(f"y_scalar: {y_scalar}")
        xp_scalar = self.xp_index[idx]
        logger.info(f"xp_scalar: {xp_scalar}")
        b_scalar = self.b_index[idx]
        logger.info(f"b_scalar: {b_scalar}")
        y = self.get_onehot_cell_types(y_scalar)
        xp = self.get_onehot_perturbs(xp_scalar)
        b = self.get_onehot_batches(b_scalar)

        # raw number from unique index array, for lookup in control_lookup
        is_control = self.perturb_flattened[idx] == self.control_label

        # Pick a control matching the perturbation
        if len(self.control_lookup[(y_scalar, b_scalar)]) > 0:
            ctrl_idx = np.random.choice(self.control_lookup[(y_scalar, b_scalar)])
            ctrl_barcode = self.cell_barcodes_flattened[ctrl_idx]
            x_ctrl_matched = adata.get([ctrl_barcode])
            x_ctrl_matched = torch.tensor(x_ctrl_matched, dtype=torch.float32, device=self.device)
        else:
            x_ctrl_matched = x.clone()
        return x, y, b, xp, x_ctrl_matched
     

if __name__ == "__main__":
    import glob
    h5ad_files = glob.glob("/home/tphan/state/state/competition_support_set/*.h5")
    ds = PerturbDataset(h5ad_files)
    x, y, b, xp, x_ctrl_matched = ds[0]
    print(x.shape)
    print(y.shape)
    print(b.shape)
    print(xp.shape)
    print(x_ctrl_matched.shape)