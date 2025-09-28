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


# isort: dont-add-import: from __future__ import annotations
from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

from daft import Expression
from daft.daft import PyPartitionField, PyPushdowns, PyRecordBatch, ScanTask
from daft.io.scan import ScanOperator
from daft.logical.schema import Schema
from daft.recordbatch import RecordBatch

if TYPE_CHECKING:
    import pyarrow
    import pyarrow.compute

try:
    import oxbow
except ImportError:
    raise ImportError("oxbow is required for VCF support. Install with: pip install oxbow")  # noqa B904

import logging

logger = logging.getLogger(__name__)
# log to file
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler("vcf_scan.log"))


def _vcf_table_factory_function(
    file_path: str,
    required_columns: list[str] | None = None,
    arrow_filters: pyarrow.compute.Expression | None = None,
    info_fields: list[str] | None = None,
    samples: list[str] | None = None,
    genotype_fields: list[str] | None = None,
    regions: list[str] | None = None,
    batch_size: int = 131072,
) -> Iterator[PyRecordBatch]:
    """A factory function that reads a VCF file using oxbow,
    and returns an iterator of Daft RecordBatches."""

    # Create VcfFile with appropriate parameters
    vcf_kwargs = {"batch_size": batch_size}

    if info_fields is not None:
        vcf_kwargs["info_fields"] = info_fields
    if samples is not None:
        vcf_kwargs["samples"] = samples
    if genotype_fields is not None:
        vcf_kwargs["genotype_fields"] = genotype_fields
    if regions is not None:
        vcf_kwargs["regions"] = regions

    vcf_file = oxbow.from_vcf(file_path, **vcf_kwargs)

    # Generate batches
    for batch in vcf_file.batches():
        # Convert dictionary columns to regular string columns (Daft doesn't support dictionary types)
        import pyarrow as pa

        columns_to_cast = []
        new_columns = []

        for i, field in enumerate(batch.schema):
            if pa.types.is_dictionary(field.type):  # Dictionary type
                # Somehow oxbow stores the chrom column as a dictionary
                # -- dictionary:
                # ["19", "20", "Y", "HLA-A*01:01:01:01"]
                # -- indices:
                # [0, 1, 1, 1, 1]
                # Cast dictionary to string
                logger.debug(f"Casting dictionary column {field.name} to string. {batch['chrom']}")
                new_columns.append(batch.column(i).dictionary_decode())
                columns_to_cast.append(field.name)
            else:
                new_columns.append(batch.column(i))

        if columns_to_cast:
            # Create new schema with string types for dictionary columns
            new_fields = []
            for field in batch.schema:
                if pa.types.is_dictionary(field.type):  # Dictionary type
                    new_fields.append(pa.field(field.name, pa.string()))
                else:
                    new_fields.append(field)
            new_schema = pa.schema(new_fields)
            batch = pa.record_batch(new_columns, schema=new_schema)

        # Apply column filtering if needed
        if required_columns is not None:
            # Filter to only the required columns that exist in the batch
            available_columns = [col for col in required_columns if col in batch.schema.names]
            if available_columns:
                batch = batch.select(available_columns)

        # Apply row filters if needed
        if arrow_filters is not None:
            batch = batch.filter(arrow_filters)

        yield RecordBatch.from_arrow_record_batches([batch], batch.schema)._recordbatch


class VCFScanOperator(ScanOperator):
    def __init__(
        self,
        path: str,
        info_fields: list[str] | None = None,
        samples: list[str] | None = None,
        genotype_fields: list[str] | None = None,
        regions: list[str] | None = None,
        batch_size: int = 131072,
    ):
        super().__init__()
        self._path = path
        self._info_fields = info_fields
        self._samples = samples
        self._genotype_fields = genotype_fields
        self._regions = regions
        self._batch_size = batch_size

        if not os.path.isfile(path):
            raise FileNotFoundError(f"VCF file not found: {path}")

        # Create a VcfFile instance to get schema and metadata
        vcf_kwargs = {}
        if info_fields is not None:
            vcf_kwargs["info_fields"] = info_fields
        if samples is not None:
            vcf_kwargs["samples"] = samples
        if genotype_fields is not None:
            vcf_kwargs["genotype_fields"] = genotype_fields
        if regions is not None:
            vcf_kwargs["regions"] = regions

        self._vcf_file = oxbow.from_vcf(path, **vcf_kwargs)

        # Convert dictionary types to string types for schema (Daft doesn't support dictionary types)
        import pyarrow as pa

        original_schema = self._vcf_file.schema
        new_fields = []
        for field in original_schema:
            if pa.types.is_dictionary(field.type):  # Dictionary type
                new_fields.append(pa.field(field.name, pa.string()))
            else:
                new_fields.append(field)

        compatible_schema = pa.schema(new_fields)
        self._schema = Schema.from_pyarrow_schema(compatible_schema)

    def name(self) -> str:
        return "VCFScanOperator"

    def display_name(self) -> str:
        return f"VCFScanOperator({self._path})"

    def schema(self) -> Schema:
        return self._schema

    def partitioning_keys(self) -> list[PyPartitionField]:
        return []

    def can_absorb_filter(self) -> bool:
        return True

    def can_absorb_limit(self) -> bool:
        return False

    def can_absorb_select(self) -> bool:
        return True

    def multiline_display(self) -> list[str]:
        display_lines = [
            self.display_name(),
            f"Schema = {self.schema()}",
        ]

        # Add information about samples if available
        if hasattr(self._vcf_file, "samples") and self._vcf_file.samples:
            display_lines.append(f"Samples = {len(self._vcf_file.samples)}")

        # Add information about regions if specified
        if self._regions:
            display_lines.append(f"Regions = {len(self._regions)}")

        return display_lines

    def to_scan_tasks(self, pushdowns: PyPushdowns) -> Iterator[ScanTask]:
        required_columns = pushdowns.columns
        filter = pushdowns.filters
        arrow_filters = None
        # TODO: if there is a filter column, the to-read columns must include filter columns and pushdown columns
        if filter is not None:
            arrow_filters = Expression._from_pyexpr(filter).to_arrow_expr()

        # TODO: by default only load up to a number of info fields
        # For VCF files, we typically create a single scan task per file
        # TODO: More sophisticated partitioning could be added based on regions
        # TODO: We can use a pre-defined partition scheme for human genome
        yield ScanTask.python_factory_func_scan_task(
            module=_vcf_table_factory_function.__module__,
            func_name=_vcf_table_factory_function.__name__,
            func_args=(
                self._path,
                required_columns,
                arrow_filters,
                self._info_fields,
                self._samples,
                self._genotype_fields,
                self._regions,
                self._batch_size,
            ),
            schema=self._schema._schema,
            num_rows=None,
            size_bytes=None,
            pushdowns=pushdowns,
            stats=None,
        )
