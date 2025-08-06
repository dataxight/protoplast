import logging
import sys
from collections.abc import Iterator
import time
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute
import pyarrow.compute as pc
from daft import Expression
from daft.daft import PyPartitionField, PyPushdowns, PyRecordBatch, ScanTask, PyField, PyDataType
from daft.io.scan import ScanOperator
from daft.logical.schema import Schema
from daft.recordbatch import RecordBatch
from pyarrow import RecordBatch as pa_RecordBatch

logger = logging.getLogger(__name__)
# log to file, with timestamp formatted as YYYY-MM-DD HH:MM:SS, and log to console
logger.setLevel(logging.DEBUG)

# Create formatter with timestamp and function line information
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create and configure file handler
file_handler = logging.FileHandler("anndata_scan.log", mode="a")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create and configure console handler
console_handler = logging.StreamHandler(sys.stdout)
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
        self._matrix_info = None
        self._n_cells = None
        self.var_h5dataset: str = var_h5dataset


    @property
    def gene_names(self) -> np.ndarray:
        """Cached gene names - only read once"""
        if self._gene_name_cache is None:
            with h5py.File(self.filename, "r", locking=False) as f:
                gene_names = f[self.var_h5dataset][:]
                gene_names = [gene.decode("utf-8") for gene in gene_names]
                self._gene_name_cache = gene_names
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
        pyarrow_schema = pa.schema([pa.field(gene_name, pa.float32()) for gene_name in gene_names] + [pa.field("__cell_id", pa.int32())])
        return Schema.from_pyarrow_schema(pyarrow_schema)

    @property
    def matrix_info(self) -> dict:
        """Cache matrix metadata to avoid repeated file access"""
        if self._matrix_info is None:
            with h5py.File(self.filename, "r", locking=False) as f:
                X_group = f["X"]
                if isinstance(X_group, h5py.Group):
                    self._matrix_info = {
                        "format": "sparse",
                        "shape": tuple(X_group.attrs["shape"]),
                        "nnz": len(X_group["data"]),
                        "dtype": X_group["data"].dtype,
                    }
                else:
                    self._matrix_info = {"format": "dense", "shape": X_group.shape, "dtype": X_group.dtype}
        return self._matrix_info

    @property
    def n_cells(self) -> int:
        """Cached number of cells - only read once"""
        if self._n_cells is None:
            self._n_cells = self.matrix_info["shape"][0]
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

    def read_cells_data_to_record_batch2(
        self, start_idx: int, end_idx: int, schema: Schema, pyarrow_filters: pyarrow.compute.Expression | None = None
    ) -> Iterator[PyRecordBatch]:
        """
        Read a batch of cells into memory
        """
        # TODO: use the correct dtype. Should use int32 for count data, float32 for other data
        start_time = time.time()
        batch_data = {field.name: np.zeros(end_idx - start_idx, dtype=np.float32) for field in schema}
        batch_data["__cell_id"] = pa.array(np.arange(start_idx, end_idx, dtype=np.int32))
        # TODO: as we can provide the file-like object, consider using fsspec so we can read it remotely
        with h5py.File(self.filename, "r", locking=False) as f:
            cells = f["X"]["indptr"][start_idx : end_idx + 1]  # +1 because we want to include where the last cell ends
            read_start = cells[0]
            read_end = cells[-1]
            z = pa.array(f["X"]["data"][read_start:read_end])
            y = pa.array(f["X"]["indices"][read_start:read_end], type=pa.int32())
            x = np.zeros(read_end - read_start, dtype=np.int32)
            logger.debug(f"Reading cells {start_idx} to {end_idx} in {time.time() - start_time} seconds")

            # iterate the cells and read the data
            for i, cell_start in enumerate(cells[:-1]):  # -1 because the last value is not a cell start
                cell_end = cells[i + 1]
                cell_start -= read_start  # adjust the coordinates to the read start
                cell_end -= read_start  # adjust the coordinates to the read start
                x[cell_start:cell_end] = i  # fill the array with the cell index
            x = pa.array(x)
            ccr = pa.table({"x": x, "y": y, "z": z})

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

        # convert the batch data to a pyarrow record batch, zero-copy
        batch = pa_RecordBatch.from_pydict(batch_data)
        if pyarrow_filters is not None:
            batch = batch.filter(pyarrow_filters)
        logger.debug(f"End reading cells {start_idx} to {end_idx} in {time.time() - start_time} seconds")
        yield RecordBatch.from_arrow_record_batches([batch], batch.schema)._recordbatch


def _h5ad_data__factory_function(
    file_path: str,
    var_h5dataset: str,
    start_idx: int,
    end_idx: int,
    schema: Schema,
    arrow_filters: pyarrow.compute.Expression | None = None,
) -> Iterator[PyRecordBatch]:
    """A factory function that reads a single CSV file into pyarrow,
    and returns an iterator of Daft RecordBatches."""

    reader = H5ADReader(file_path, var_h5dataset)
    return reader.read_cells_data_to_record_batch2(start_idx, end_idx, schema, arrow_filters)


class H5ADScanOperator(ScanOperator):
    def __init__(self, path: str, batch_size: int = 10000, var_h5dataset: str = "var/_index"):
        super().__init__()
        self._path = path
        self._reader = H5ADReader(path, var_h5dataset)
        self._schema = self._reader.create_schema_from_genes()
        self._batch_size = batch_size
        self._var_h5dataset = var_h5dataset

    def name(self) -> str:
        return "H5ADScanOperator"

    def display_name(self) -> str:
        return f"H5ADScanOperator({self._path})"

    def schema(self) -> Schema:
        return self._schema

    def partitioning_keys(self) -> list[PyPartitionField]:
        return [PyPartitionField(PyField.create("__cell_id", PyDataType.int32()))]

    def can_absorb_filter(self) -> bool:
        return True

    def can_absorb_limit(self) -> bool:
        return True

    def can_absorb_select(self) -> bool:
        return True

    def multiline_display(self) -> list[str]:
        return [
            self.display_name(),
            f"Schema = {self.schema()}",
            f"Num cells = {self._reader.n_cells}",
        ]

    def to_scan_tasks(self, pushdowns: PyPushdowns) -> Iterator[ScanTask]:
        # The maximum possible columns we need to read is the projection columns + the filter columns
        to_read_columns: list[str] | None
        filter_required_column_names = pushdowns.filter_required_column_names()

        if pushdowns.columns is None:
            after_scan_columns = []
        else:
            after_scan_columns = pushdowns.columns

        # if no columns are specified, read up to 20 genes
        if not after_scan_columns:
            # if no columns are specified, read up to 20 genes
            n_genes = min(20, len(self._reader.gene_names))
            after_scan_columns = self._reader.gene_names[:n_genes]

        # include the filter required columns to the required columns
        if filter_required_column_names is not None:
            to_read_columns = [c for c in set(after_scan_columns + filter_required_column_names)]
        else:
            to_read_columns = after_scan_columns

        # create the schema for the pushdown
        push_down_schema = self._reader.create_schema_from_genes(to_read_columns)
        after_scan_schema = self._reader.create_schema_from_genes(after_scan_columns)

        filter = pushdowns.filters
        arrow_filters = None
        if filter is not None:
            arrow_filters = Expression._from_pyexpr(filter).to_arrow_expr()

        batches = [(start_idx, end_idx) for start_idx, end_idx in self._reader.generate_cell_batches(self._batch_size)]
        for start_idx, end_idx in batches:
            yield ScanTask.python_factory_func_scan_task(
                module=_h5ad_data__factory_function.__module__,
                func_name=_h5ad_data__factory_function.__name__,
                func_args=(self._path, self._var_h5dataset, start_idx, end_idx, push_down_schema, arrow_filters),
                schema=after_scan_schema._schema,
                num_rows=None,
                size_bytes=None,
                pushdowns=pushdowns,
                stats=None,
            )
