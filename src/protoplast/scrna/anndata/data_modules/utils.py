import logging
from pathlib import Path

import pickle
from typing import Dict, List, Optional, Tuple

import anndata
import h5py
import numpy as np
import toml
import torch
from braceexpand import braceexpand
import pandas as pd
import scipy.sparse
from anndata._core.sparse_dataset import sparse_dataset

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(log_fmt))
logger.addHandler(handler)

def make_onehot_encoding_map(labels):
    labels = list(labels)  # make sure it's a list
    n = len(labels)
    return {
        label: torch.tensor(np.eye(n, dtype=np.float32)[i], dtype=torch.float32)
        for i, label in enumerate(labels)
    }

def parse_dataset_config(config_path: str) -> dict:
    """
    Parse TOML config file for dataset configuration.

    Returns:
        Dictionary with parsed config including file paths and split assignments
    """
    config = toml.load(config_path)

    # Expand dataset paths using glob patterns
    expanded_datasets = {}
    for dataset_name, pattern in config["datasets"].items():
        files = list(braceexpand(pattern))
        if not files:
            logger.warning(f"No files found for dataset {dataset_name} with pattern {pattern}")
            continue
        expanded_datasets[dataset_name] = files

    # Create file to split mapping and per-target splits
    file_splits = {}
    target_splits = {}  # {(dataset, cell_type): {target: split}}

    # Parse fewshot rules for per-target splits
    if "fewshot" in config:
        for fewshot_key, split_rules in config["fewshot"].items():
            # Extract dataset and cell type from key like "replogle_h1.ARC_H1"
            if "." in fewshot_key:
                dataset_name, cell_type = fewshot_key.split(".", 1)
                target_splits[(dataset_name, cell_type)] = {}

                # Parse split rules like {"val": ["target1", "target2"], "test": ["target3"]}
                for split_name, targets in split_rules.items():
                    for target in targets:
                        target_splits[(dataset_name, cell_type)][target] = split_name

    # Process each dataset
    for dataset_name, files in expanded_datasets.items():
        for file_path in files:
            # Extract cell type from filename (assuming format like k562.h5, rpe1.h5, etc.)
            cell_type = Path(file_path).stem

            # Default to train
            split = "train"

            # Check zeroshot rules (entire cell types go to val/test)
            zeroshot_key = f"{dataset_name}.{cell_type}"
            if "zeroshot" in config and zeroshot_key in config["zeroshot"]:
                split = config["zeroshot"][zeroshot_key]

            file_splits[file_path] = {
                "split": split,
                "dataset": dataset_name,
                "cell_type": cell_type
            }

    return {
        "file_splits": file_splits,
        "target_splits": target_splits,
        "config": config
    }


def compute_and_save_target_means(
    files: List[str],
    tensor_output_path: str,
    index_output_path: str,
    *,
    target_label: str = "target_gene",
    cell_type_label: str = "cell_type",
    control_label: Optional[str] = None,
    sparse_key: str = "X",
) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute per-target mean expression across all files and save a single tensor and address list.

    Args:
        files: List of .h5ad paths. Each file contains a single cell type, with
               cells sorted by target gene.
        tensor_output_path: Path to save the stacked tensor with torch.save.
                            Saved tensor shape: torch.Tensor [n_targets, G], where n_targets
                            is the number of unique targets (excluding control), averaged
                            across all cell types.
        index_output_path: Path to save the address list with pickle.
                            Saved object: List[str] of length n_targets with entries
                            "<target>" corresponding to rows of the tensor.
        target_label: obs column name containing target gene label.
        cell_type_label: obs column name containing cell type label.
        control_label: If provided, skip this label from targets (e.g., "non-targeting").
        sparse_key: Key for expression matrix in h5ad (usually "X").

    Returns:
        Tuple of (tensor, addresses):
            - tensor: torch.Tensor with shape [n_targets, G]
            - addresses: List[str] with length n_targets, where addresses[i] == "<target>"
    """
    logger.info(f"Computing target means for {len(files)} files")

    cell_type_to_means: Dict[str, Dict[str, torch.Tensor]] = {}
    global_num_genes: Optional[int] = None

    for file_i,file_path in enumerate(files):
        logger.info(f"Processing file: {file_path} ({file_i+1}/{len(files)})")
        adata = anndata.read_h5ad(file_path, backed="r")
        obs = adata.obs

        if obs.shape[0] == 0:
            logger.warning(f"Empty AnnData: {file_path}")
            continue

        # Expect exactly one cell type per file
        file_cell_type = str(obs[cell_type_label].iloc[0])

        # Determine gene dimension
        num_genes = adata.n_vars
        if global_num_genes is None:
            global_num_genes = num_genes
        elif num_genes != global_num_genes:
            raise ValueError(
                f"Gene dimension mismatch across files: got {num_genes} for {file_path}, expected {global_num_genes}"
            )

        # Run-length encode contiguous target groups
        targets = obs[target_label].values
        change_points = np.where(targets[1:] != targets[:-1])[0] + 1
        starts = np.r_[0, change_points]
        ends = np.r_[change_points, len(targets)]
        target_names_in_order = [str(targets[s]) for s in starts]

        # Initialize containers if first encounter of this cell type
        if file_cell_type not in cell_type_to_means:
            cell_type_to_means[file_cell_type] = {}

        # Open underlying h5 file to read sparse rows efficiently
        with h5py.File(file_path, "r", libver="latest") as h5f:
            X_ds = sparse_dataset(h5f[sparse_key])

            for target_name, start, end in zip(target_names_in_order, starts, ends):
                if control_label is not None and target_name == control_label:
                    continue

                X_block = X_ds[start:end]
                if not scipy.sparse.issparse(X_block):
                    X_block = scipy.sparse.csr_matrix(X_block)

                # Compute mean across cells
                # Using sum/num to avoid densifying
                num_cells = max(1, (end - start))
                sum_vec = np.asarray(X_block.sum(axis=0)).ravel()
                mean_vec = (sum_vec / float(num_cells)).astype(np.float32)

                # Sanity on gene dimension
                if mean_vec.shape[0] != num_genes:
                    raise ValueError(
                        f"Gene dimension mismatch in {file_path}: got {mean_vec.shape[0]}, expected {num_genes}"
                    )
                if target_name not in cell_type_to_means[file_cell_type]:
                    cell_type_to_means[file_cell_type][target_name] = torch.from_numpy(mean_vec)
                else:
                    new_mean = torch.stack([
                        cell_type_to_means[file_cell_type][target_name],
                        torch.from_numpy(mean_vec),
                    ]).mean(dim=0)
                    cell_type_to_means[file_cell_type][target_name] = new_mean

    # Aggregate across cell types: build per-target means averaged over cell types
    target_to_vecs: Dict[str, List[torch.Tensor]] = {}
    for ct in sorted(cell_type_to_means.keys()):
        mean_dict = cell_type_to_means[ct]
        if len(mean_dict) == 0:
            logger.warning(f"No targets collected for cell type {ct}")
            continue
        for target_name, vec in mean_dict.items():
            if target_name not in target_to_vecs:
                target_to_vecs[target_name] = []
            target_to_vecs[target_name].append(vec)

    # Build a single stacked tensor [n_targets, G] and the corresponding addresses list
    addresses: List[str] = []
    rows: List[torch.Tensor] = []
    for target_name in sorted(target_to_vecs.keys()):
        ct_stack = torch.stack(target_to_vecs[target_name], dim=0)
        rows.append(ct_stack.mean(dim=0))
        addresses.append(target_name)

    if global_num_genes is None:
        global_num_genes = 0
    data_tensor: torch.Tensor
    if len(rows) == 0:
        data_tensor = torch.empty((0, global_num_genes), dtype=torch.float32)
    else:
        data_tensor = torch.stack(rows, dim=0)

    # Save outputs
    torch.save(data_tensor, tensor_output_path)
    with open(index_output_path, "wb") as f:
        pickle.dump(addresses, f)

    logger.info(f"Saved stacked target mean tensor to {tensor_output_path}")
    logger.info(f"Saved address list to {index_output_path}")

    return data_tensor, addresses

if __name__ == "__main__":
    import glob
    files = glob.glob("/mnt/hdd2/tan/competition_support_set_sorted/competition_train.h5")
    tensor_output_path = "./competition_support_mean.torch"
    index_output_path = "./competition_support_mean_index.pkl"
    compute_and_save_target_means(files, tensor_output_path, index_output_path)
