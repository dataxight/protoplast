import torch
import numpy as np

def make_onehot_encoding_map(labels):
    labels = list(labels)  # make sure it's a list
    n = len(labels)
    return {
        label: torch.tensor(np.eye(n, dtype=np.float32)[i], dtype=torch.float32)
        for i, label in enumerate(labels)
    }