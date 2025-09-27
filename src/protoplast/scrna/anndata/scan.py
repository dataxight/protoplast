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

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anndata
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from daft.io.source import DataSource, DataSourceTask
from daft.logical.schema import Schema
from daft.recordbatch.micropartition import MicroPartition

from protoplast.utils import ExpressionVisitorWithRequiredColumns

if TYPE_CHECKING:
    from daft.io import IOConfig
    from daft.io.pushdowns import Pushdowns


logger = logging.getLogger(__name__)
# log to file
logger.setLevel(logging.INFO)
# format the logger handler to include the timestamp
handler = logging.FileHandler("anndata_scan.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# add a console handler to the logger
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class H5ADReader:
    """
    Reads metadata from an h5ad file.

    Args:
        filename (str): The path to the h5ad file.

    Returns:
        MetadataReader: A MetadataReader object.
    """

    def __init__(self, filename: str, var_h5dataset: str = "var/_index"):
        self.filename = filename
        self._gene_name_cache = None
        self._gene_index_map = None
        self._n_cells = None
        self.var_h5dataset: str = var_h5dataset
        self._ad = anndata.read_h5ad(filename, backed="r")

    @property
    def gene_names(self) -> np.ndarray:
        """Cached gene names - only read once"""
        if self._gene_name_cache is None:
            self._gene_name_cache = self._ad.var_names.to_list()
        return self._gene_name_cache

    @property
    def gene_index_map(self) -> dict[str, int]:
        """Cached gene name to index mapping - O(1) lookups"""
        if self._gene_index_map is None:
            gene_names = self.gene_names
            # Use numpy's approach for fastest dict creation
            self._gene_index_map = {gene: idx for idx, gene in enumerate(gene_names)}
        return self._gene_index_map

    def create_schema_from_genes(self, gene_names: list[str] = None) -> Schema:
        """
        Creates a schema from genes. gene_names filtering is optional
        Each gene is a column in the schema.
        """
        if gene_names is None:
            gene_names = self.gene_names

        # TODO: have to use int if the data is count data
        return Schema.from_pyarrow_schema(pa.schema([pa.field(gene_name, pa.float32()) for gene_name in gene_names]))

    @property
    def n_cells(self) -> int:
        """Cached number of cells - only read once"""
        if self._n_cells is None:
            self._n_cells = self._ad.X.shape[0]
        return self._n_cells

    def generate_cell_batches(self, batch_size: int) -> Iterator[tuple[int, int]]:
        """
        Calculates the number of cells to read in each batch.
        """
        # get n cells from the matrix_info
        n = self.n_cells
        max_batch_size = min(batch_size, n)
        for i in range(0, n, max_batch_size):
            start_row = i
            end_row = min(i + max_batch_size, n)
            yield start_row, end_row

    def read_cells_data_to_micropartition2(
        self,
        start_idx: int,
        end_idx: int,
        schema: Schema,
        after_scan_schema: Schema,
        pyarrow_filters: pc.Expression | None = None,
        dtype: np.dtype = np.float32,
    ) -> MicroPartition:
        """
        Read a batch of cells into memory and return a MicroPartition
        """
        logger.debug(f"Reading cells from {start_idx} to {end_idx}")

        ad = anndata.read_h5ad(self.filename, backed="r")
        # rows are cells, columns are genes
        sparse_X = ad.X._to_backed()
        logger.debug(f"reading sparse_X: {start_idx} to {end_idx}")
        logger.debug(f"dtype: {ad.X.dtype}")
        gene_indices = [self.gene_index_map[gene] for gene in after_scan_schema.column_names()]
        # NOTE: somehow the dtype loaded by anndata is float64 in case of float data
        dense_X = sparse_X[start_idx:end_idx, gene_indices].toarray()
        logger.debug(f"Densified. {start_idx} to {end_idx} {dense_X.shape}")
        dense_T = dense_X.T
        logger.debug(f"Transposed. {start_idx} to {end_idx} {dense_T.shape}")
        # TODO: use the correct dtype. Should use int32 for count data, float32 for other data
        batch_data = [pa.array(row_t, type=pa.float32()) for row_t in dense_T]
        logger.debug(f"Batch data. {start_idx} to {end_idx}")
        batch = pa.RecordBatch.from_arrays(batch_data, names=[gene for gene in after_scan_schema.column_names()])
        logger.debug(f"Batch. {start_idx} to {end_idx} {batch.num_rows}")
        if pyarrow_filters is not None:
            batch = batch.filter(pyarrow_filters)
        logger.debug(f"Filtered. {start_idx} to {end_idx} {batch.num_rows}")
        return MicroPartition.from_arrow_record_batches([batch], arrow_schema=after_scan_schema.to_pyarrow_schema())

    def read_cells_data_to_micropartition(
        self,
        start_idx: int,
        end_idx: int,
        schema: Schema,
        after_scan_schema: Schema,
        pyarrow_filters: pc.Expression | None = None,
        dtype: np.dtype = np.float32,
    ) -> MicroPartition:
        """
        Read a batch of cells into memory and return a MicroPartition
        """
        # TODO: use the correct dtype. Should use int32 for count data, float32 for other data
        logger.info(f"Reading cells from {start_idx} to {end_idx}")
        batch_data = {field.name: np.zeros(end_idx - start_idx, dtype=np.float32) for field in schema}
        logger.debug(f"Init zero batch data {start_idx} to {end_idx} ")

        # TODO: as we can provide the file-like object, consider using fsspec so we can read it remotely
        with h5py.File(self.filename, "r") as f:
            cells = f["X"]["indptr"][start_idx : end_idx + 1]  # +1 because we want to include where the last cell ends
            read_start = cells[0]
            read_end = cells[-1]
            z = pa.array(f["X"]["data"][read_start:read_end])
            y = pa.array(f["X"]["indices"][read_start:read_end], type=pa.int32())
            x = np.zeros(read_end - read_start, dtype=np.int32)
            logger.debug(f"Z. {start_idx} to {end_idx} ")

            # iterate the cells and read the data
            for i, cell_start in enumerate(cells[:-1]):  # -1 because the last value is not a cell start
                cell_end = cells[i + 1]
                cell_start -= read_start  # adjust the coordinates to the read start
                cell_end -= read_start  # adjust the coordinates to the read start
                x[cell_start:cell_end] = i  # fill the array with the cell index
            x = pa.array(x)
            ccr = pa.table({"x": x, "y": y, "z": z})
            logger.debug(f"CCRC. {start_idx} to {end_idx} {ccr.num_rows}")

            column_names = schema.column_names()
            # get the indices of the projection genes
            yy = [i for i, gene_name in enumerate(self.gene_names) if gene_name in column_names]
            # only keep the genes that are in the schema
            ccr = ccr.filter(pc.is_in(ccr["y"], pa.array(yy)))
            for yi in yy:
                # for each gene, get the mask of the cells that have non-zero values
                # then set the values in the batch data
                non_zero_cell_indices = ccr.filter(pc.equal(ccr["y"], yi))["x"].to_numpy()
                batch_data[self.gene_names[yi]][non_zero_cell_indices] = ccr.filter(pc.equal(ccr["y"], yi))[
                    "z"
                ].to_numpy()
            logger.debug(f"Batch data. {start_idx} to {end_idx} ")

        # convert the batch data to a pyarrow record batch, zero-copy
        batch_data = {field.name: batch_data[field.name] for field in after_scan_schema}
        batch = pa.RecordBatch.from_pydict(batch_data)
        logger.debug(f"Batch. {start_idx} to {end_idx} {batch.num_rows}")
        if pyarrow_filters is not None:
            batch = batch.filter(pyarrow_filters)
        logger.debug(f"Filtered. {start_idx} to {end_idx} {batch.num_rows}")
        return MicroPartition.from_arrow_record_batches([batch], arrow_schema=after_scan_schema.to_pyarrow_schema())


class H5ADSource(DataSource):
    def __init__(
        self,
        file_path: str,
        batch_size: int = 10000,
        preview_size: int = 20,
        var_h5dataset: str = "var/_index",
        io_config: IOConfig | None = None,
    ):
        self._file_path = file_path
        self._batch_size = batch_size
        self._var_h5dataset = var_h5dataset
        self._io_config = io_config
        self._reader = H5ADReader(file_path, var_h5dataset)
        if preview_size == 0:
            self._schema = self._reader.create_schema_from_genes()
            self._n_genes = len(self._reader.gene_names)
        else:
            self._schema = self._reader.create_schema_from_genes(self._reader.gene_names[:preview_size])
            self._n_genes = preview_size

    @property
    def name(self) -> str:
        return "H5ADSource"

    @property
    def schema(self) -> Schema:
        return self._schema

    def display_name(self) -> str:
        return f"H5ADSource({self._file_path})"

    def multiline_display(self) -> list[str]:
        return [
            self.display_name(),
            f"Schema = {self._schema}",
            f"Num cells = {self._reader.n_cells}",
        ]

    def get_tasks(self, pushdowns: Pushdowns | None = None) -> Iterator[H5ADSourceTask]:
        # The maximum possible columns we need to read is the projection columns + the filter columns
        to_read_columns: list[str] | None
        after_scan_columns: list[str] = []
        visitor = ExpressionVisitorWithRequiredColumns()

        if pushdowns is not None:
            if pushdowns.filters is not None:
                filter_required_column_names = visitor.get_required_columns(pushdowns.filters)
            else:
                filter_required_column_names = []

            if pushdowns.columns is not None:
                after_scan_columns = pushdowns.columns

            # if no columns are specified, read up to 20 genes
            if not after_scan_columns:
                after_scan_columns = self._reader.gene_names[: self._n_genes]

            # include the filter required columns (if any)to the required columns
            to_read_columns = list(set(after_scan_columns + filter_required_column_names))
        else:
            # Default case when no pushdowns
            after_scan_columns = self._reader.gene_names[: self._n_genes]
            to_read_columns = after_scan_columns

        # create the schema for the pushdown
        push_down_schema = self._reader.create_schema_from_genes(to_read_columns)
        after_scan_schema = self._reader.create_schema_from_genes(after_scan_columns)

        arrow_filters = None
        if pushdowns is not None and pushdowns.filters is not None:
            arrow_filters = pushdowns.filters.to_arrow_expr()

        for start_idx, end_idx in self._reader.generate_cell_batches(self._batch_size):
            yield H5ADSourceTask(
                _file_path=self._file_path,
                _var_h5dataset=self._var_h5dataset,
                _start_idx=start_idx,
                _end_idx=end_idx,
                _push_down_schema=push_down_schema,
                _after_scan_schema=after_scan_schema,
                _arrow_filters=arrow_filters,
                _io_config=self._io_config,
            )


@dataclass
class H5ADSourceTask(DataSourceTask):
    _file_path: str
    _var_h5dataset: str
    _start_idx: int
    _end_idx: int
    _push_down_schema: Schema
    _after_scan_schema: Schema
    _arrow_filters: pc.Expression | None = None
    _io_config: IOConfig | None = None

    def execute(self) -> Iterator[MicroPartition]:
        reader = H5ADReader(self._file_path, self._var_h5dataset)
        micropartition = reader.read_cells_data_to_micropartition2(
            self._start_idx, self._end_idx, self._push_down_schema, self._after_scan_schema, self._arrow_filters
        )

        yield micropartition

    def get_micro_partitions(self) -> Iterator[MicroPartition]:
        yield from self.execute()

    @property
    def schema(self) -> Schema:
        return self._after_scan_schema
