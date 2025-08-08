# ruff: noqa: I002
# isort: dont-add-import: from __future__ import annotations

from __future__ import annotations

from typing import TYPE_CHECKING

from daft.api_annotations import PublicAPI

from .scan import H5ADSource

if TYPE_CHECKING:
    from daft import DataFrame
    from daft.io import IOConfig


@PublicAPI
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
