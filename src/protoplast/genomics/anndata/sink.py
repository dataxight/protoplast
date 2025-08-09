from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Iterator

import daft
import pyarrow as pa
import pyarrow.parquet as pq
from daft.io.sink import DataSink, WriteResult
from daft.recordbatch import MicroPartition

if TYPE_CHECKING:
    from daft.schema import Schema


class AnnDataSink(DataSink[str]):
    """A DataSink that writes micropartitions to parquet files in a specified directory."""

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def name(self) -> str:
        return f"AnnDataSink({self._path})"

    def schema(self) -> Schema:
        """The schema of the MicroPartition returned by finalize."""
        return daft.Schema.from_pydict({"path": daft.DataType.string()})

    def start(self) -> None:
        """Create the output directory."""
        # TODO: use fsspec to create the directory
        os.makedirs(self._path, exist_ok=True)

    def write(self, micropartitions: Iterator[MicroPartition]) -> Iterator[WriteResult[str]]:
        """Write each MicroPartition to a separate Parquet file."""
        for mp in micropartitions:
            if len(mp) == 0:
                continue

            table: pa.Table = mp.to_arrow()

            filename = f"{uuid.uuid4()}.parquet"
            filepath = os.path.join(self._path, filename)

            pq.write_table(table, filepath)

            bytes_written = os.path.getsize(filepath)
            rows_written = len(mp)

            yield WriteResult(
                result=filepath,
                bytes_written=bytes_written,
                rows_written=rows_written,
            )

    def finalize(self, write_results: list[WriteResult[str]]) -> MicroPartition:
        """Collects the file paths of written parquet files into a MicroPartition."""
        paths = [r.result for r in write_results]

        if not paths:
            return MicroPartition.empty(self.schema())

        return MicroPartition.from_pydict({"path": paths})
