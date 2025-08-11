from __future__ import annotations

import h5py
import fsspec
import uritools
from typing import Literal
from os import PathLike

from anndata._io.h5ad import read_dataframe, read_elem, _clean_uns, _read_raw
from anndata._core.anndata import AnnData


def read_h5ad_backed(
    filename: str | PathLike[str], mode: Literal["r", "r+"]
) -> "AnnData":
    # NOTE: I renamed from d to _d to avoid name conflict with a reserved word in pdb, used for debugging
    _d = dict(filename=filename, filemode=mode)
    f = h5py.File(fsspec.open(filename, mode="rb").open(), 'r')

    attributes = ["obsm", "varm", "obsp", "varp", "uns", "layers"]
    df_attributes = ["obs", "var"]

    if "encoding-type" in f.attrs:
        attributes.extend(df_attributes)
    else:
        for k in df_attributes:
            if k in f:  # Backwards compat
                _d[k] = read_dataframe(f[k])

    _d.update({k: read_elem(f[k]) for k in attributes if k in f})

    _d["raw"] = _read_raw(f, attrs={"var", "varm"})
    adata = AnnData(**_d)

    # Backwards compat to <0.7
    if isinstance(f["obs"], h5py.Dataset):
        _clean_uns(adata)

    return adata

def apply_read_h5ad_backed_patch():
    import anndata._io.h5ad as h5ad_module
    h5ad_module.read_h5ad_backed = read_h5ad_backed

def rollback_read_h5ad_backed_patch():
    import anndata._io.h5ad as h5ad_module
    h5ad_module.read_h5ad_backed = h5ad_module.read_h5ad_backed