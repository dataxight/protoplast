import torch
import numpy as np
import re
import glob
from typing import List

def _to_BD(t):
    # Coerce to (B, G) and contiguous
    if t.ndim == 3 and t.size(1) == 1: t = t.squeeze(1)
    if t.ndim == 3 and t.size(-1) == 1: t = t.squeeze(-1)
    if t.ndim == 1: t = t.unsqueeze(0)
    if t.ndim != 2: t = t.view(t.size(0), -1)
    return t.contiguous()

def save_checkpoint(model, optimizer, epoch, model_dir):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, f"{model_dir}/epoch={epoch}.pt")

def make_onehot_encoding_map(labels):
    labels = list(labels)  # make sure it's a list
    n = len(labels)
    return {
        label: torch.tensor(np.eye(n, dtype=int)[i])
        for i, label in enumerate(labels)
    }

def _brace_expand(pattern: str) -> List[str]:
    """
    Expand a single brace group like '/path/{a,b,c}.h5' into
    ['/path/a.h5', '/path/b.h5', '/path/c.h5'].
    Supports one brace group per pattern (simple and fast).
    """
    m = re.search(r"\{([^{}]+)\}", pattern)
    if not m:
        return [pattern]
    start, end = m.span()
    choices = m.group(1).split(",")
    head, tail = pattern[:start], pattern[end:]
    return [head + c + tail for c in choices]

def expand_globs(brace_glob_pattern: str) -> List[str]:
    files = []
    for pat in _brace_expand(brace_glob_pattern):
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    return files
