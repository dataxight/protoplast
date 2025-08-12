from __future__ import annotations
from typing import Any, Literal, Union

from daft import DataType
from daft.expressions import Expression, ExpressionVisitor
from pathlib import Path
import obstore
import fsspec
import os

REMOTE_FILE_DRIVER = "obstore"

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from obstore import ReadableFile

class ExpressionVisitorWithRequiredColumns(ExpressionVisitor[None]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_columns: set[str] = set()

    def get_required_columns(self, expr: Expression | None) -> list[str]:
        if expr is None:
            return []

        self.visit(expr)
        required_columns = list(self.required_columns)
        self.required_columns.clear()
        return required_columns

    def visit_col(self, name: str) -> None:
        self.required_columns.add(name)

    def visit_lit(self, value: Any) -> None:
        pass

    def visit_alias(self, expr: Expression, alias: str) -> None:
        self.visit(expr)

    def visit_cast(self, expr: Expression, dtype: DataType) -> None:
        self.visit(expr)

    def visit_function(self, name: str, args: list[Expression]) -> None:
        for arg in args:
            self.visit(arg)

class ObstoreReader:
    _reader: ReadableFile

    def __init__(self, reader: ReadableFile) -> None:
        self._reader = reader

    def read(self, size: int, /) -> bytes:
        return self._reader.read(size).to_bytes()

    def readall(self) -> bytes:
        return self._reader.read().to_bytes()

    def seek(self, offset: int, whence: int = 0, /):
        # TODO: Check on default for whence
        return self._reader.seek(offset, whence)

    def tell(self) -> int:
        return self._reader.tell()

def is_local_file(path: str) -> bool:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    return 'file' in fs.protocol

def get_remote_file_object(path: str, driver: Literal["fsspec", "obstore"] = "obstore") -> Any | ReadableFile:
    _fs, _, paths = fsspec.get_fs_token_paths(path)
    if 'file' in _fs.protocol:
        fq_path = "file://" + os.path.abspath(path) 
    else:
        fq_path = path
    if driver == "fsspec":
        return fsspec.open(fq_path, mode="rb").open()
    elif driver == "obstore":
        # init the store with the directory name
        dir_name = os.path.dirname(fq_path)
        store = obstore.store.from_url(dir_name)
        # TODO: make the buffer size configurable, and use the path within the store
        reader = obstore.open_reader(store, os.path.basename(fq_path), buffer_size=64 * 1024 * 1024)
        return ObstoreReader(reader)
    else:
        raise ValueError(f"Invalid driver: {driver}")