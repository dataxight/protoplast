# ruff: noqa: I002
# isort: dont-add-import: from __future__ import annotations

from daft.daft import ScanOperatorHandle
from daft.dataframe import DataFrame
from daft.logical.builder import LogicalPlanBuilder

from .scan import H5ADScanOperator


def read_h5ad(
    path: str,
    batch_size: int = 1000,
    var_h5dataset: str = "var/_index",
) -> DataFrame:
    h5ad_operator = H5ADScanOperator(path, batch_size, var_h5dataset)
    handle = ScanOperatorHandle.from_python_scan_operator(h5ad_operator)
    builder = LogicalPlanBuilder.from_tabular_scan(scan_operator=handle)
    return DataFrame(builder)
