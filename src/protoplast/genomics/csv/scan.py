# isort: dont-add-import: from __future__ import annotations
from __future__ import annotations

import glob
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pyarrow.csv as pacsv
from daft import Expression
from daft.daft import PyPartitionField, PyPushdowns, PyRecordBatch, ScanTask
from daft.io.scan import ScanOperator
from daft.logical.schema import Schema
from daft.recordbatch import RecordBatch

if TYPE_CHECKING:
    import pyarrow


def _csv_table_factory_function(
    file_path: str,
    required_columns: list[str] | None = None,
    arrow_filters: pyarrow.compute.Expression | None = None,
) -> Iterator[PyRecordBatch]:
    """A factory function that reads a single CSV file into pyarrow,
    and returns an iterator of Daft RecordBatches."""

    convert_options = pacsv.ConvertOptions()
    if required_columns:
        convert_options.include_columns = required_columns

    reader = pacsv.open_csv(file_path, convert_options=convert_options)

    while True:
        try:
            batch = reader.read_next_batch()
            if arrow_filters is not None:
                batch = batch.filter(arrow_filters)
            yield RecordBatch.from_arrow_record_batches([batch], batch.schema)._recordbatch
        except StopIteration:
            break


class CSVScanOperator(ScanOperator):
    def __init__(self, path: str):
        super().__init__()
        self._path = path
        if glob.has_magic(path) or "*" in path or "?" in path:
            self._files = sorted(glob.glob(path, recursive=True))
        elif os.path.isdir(path):
            self._files = sorted(
                [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.endswith(".csv") and os.path.isfile(os.path.join(path, f))
                ]
            )
        elif os.path.isfile(path):
            self._files = [path]
        else:
            self._files = []

        if not self._files:
            raise FileNotFoundError(f"No CSV files found at: {path}")

        # For this toy example, we'll infer the schema from the first file.
        self._schema = Schema.from_pyarrow_schema(pacsv.read_csv(self._files[0]).schema)

        # Filter out files that don't match the schema of the first file.
        self._files = [f for f in self._files if Schema.from_pyarrow_schema(pacsv.read_csv(f).schema) == self._schema]

    def name(self) -> str:
        return "CSVScanOperator"

    def display_name(self) -> str:
        return f"CSVScanOperator({self._path})"

    def schema(self) -> Schema:
        return self._schema

    def partitioning_keys(self) -> list[PyPartitionField]:
        return []

    def can_absorb_filter(self) -> bool:
        return False

    def can_absorb_limit(self) -> bool:
        return False

    def can_absorb_select(self) -> bool:
        return True

    def multiline_display(self) -> list[str]:
        return [
            self.display_name(),
            f"Schema = {self.schema()}",
            f"Num files = {len(self._files)}",
        ]

    def to_scan_tasks(self, pushdowns: PyPushdowns) -> Iterator[ScanTask]:
        required_columns = pushdowns.columns
        filter = pushdowns.filters
        arrow_filters = None
        if filter is not None:
            arrow_filters = Expression._from_pyexpr(filter).to_arrow_expr()
        print(self._schema)
        for file in self._files:
            yield ScanTask.python_factory_func_scan_task(
                module=_csv_table_factory_function.__module__,
                func_name=_csv_table_factory_function.__name__,
                func_args=(file, required_columns, arrow_filters),
                schema=self._schema._schema,
                num_rows=None,
                size_bytes=None,
                pushdowns=pushdowns,
                stats=None,
            )
