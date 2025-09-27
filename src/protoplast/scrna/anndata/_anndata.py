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


# ruff: noqa: I002
# isort: dont-add-import: from __future__ import annotations

from __future__ import annotations

from typing import TYPE_CHECKING

from .scan import H5ADSource

if TYPE_CHECKING:
    from daft import DataFrame
    from daft.io import IOConfig

# NOTE: apply the patches to the anndata module
from protoplast.patches.anndata_read_h5ad_backed import apply_read_h5ad_backed_patch
from protoplast.patches.anndata_remote import apply_file_backing_patch

apply_file_backing_patch()
apply_read_h5ad_backed_patch()


def read_h5ad(
    path: str,
    batch_size: int = 1000,
    preview_size: int = 20,
    var_h5dataset: str = "var/_index",
    io_config: IOConfig | None = None,
) -> DataFrame:
    """Read h5ad file.

    Args:
        path: h5ad file path
        batch_size: Number of cells to read in each batch.
        var_h5dataset: The h5 dataset path for variable names.
        io_config: IOConfig for the file system.

    Returns:
        DataFrame: DataFrame with the schema converted from the specified h5ad file.
    """
    return H5ADSource(
        file_path=path,
        batch_size=batch_size,
        preview_size=preview_size,
        var_h5dataset=var_h5dataset,
        io_config=io_config,
    ).read()
