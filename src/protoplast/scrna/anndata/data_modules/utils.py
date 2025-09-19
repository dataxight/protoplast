import logging
from pathlib import Path

import numpy as np
import toml
import torch
from braceexpand import braceexpand

logger = logging.getLogger(__name__)

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
