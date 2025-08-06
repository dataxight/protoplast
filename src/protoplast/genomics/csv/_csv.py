# ruff: noqa: I002
# isort: dont-add-import: from __future__ import annotations

from daft.api_annotations import PublicAPI
from daft.daft import ScanOperatorHandle
from daft.dataframe import DataFrame
from daft.logical.builder import LogicalPlanBuilder

from .scan import CSVScanOperator


@PublicAPI
def read_csv(
    path: str,
) -> DataFrame:
    """Create a DataFrame from a CSV file or directory of CSV files.

    Args:
        path (str): Path to the CSV file or directory. Can be a glob pattern.

    Returns:
        DataFrame: A DataFrame with the data from the CSV file(s).

    Examples:
        Reading a single CSV file:
        >>> df = daft.read_csv("data.csv")
        >>> df.show()

        Reading a directory of CSV files:
        >>> df = daft.read_csv("my_csv_folder/")
        >>> df.show()

        Reading CSV files using a glob pattern:
        >>> df = daft.read_csv("my_csv_folder/*.csv")
        >>> df.show()
    """
    csv_operator = CSVScanOperator(path)
    handle = ScanOperatorHandle.from_python_scan_operator(csv_operator)
    builder = LogicalPlanBuilder.from_tabular_scan(scan_operator=handle)
    return DataFrame(builder)
