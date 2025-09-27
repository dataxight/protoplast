#   Copyright 2025 DataXight, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


from __future__ import annotations

from os import PathLike
from typing import Literal

import h5py
from anndata._core.anndata import AnnData
from anndata._io.h5ad import _clean_uns, _read_raw, read_dataframe, read_elem

from .file_handler import open_fsspec


def read_h5ad_backed(filename: str | PathLike[str], mode: Literal["r", "r+"]) -> AnnData:
    # NOTE: I renamed from d to _d to avoid name conflict with a reserved word in pdb, used for debugging
    _d = dict(filename=filename, filemode=mode)
    f = open_fsspec(filename)

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
