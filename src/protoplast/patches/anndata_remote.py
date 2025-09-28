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

from os import PathLike
from pathlib import Path
from typing import Literal

import fsspec
import h5py
from anndata._core.file_backing import AnnDataFileManager as OriginalAnnDataFileManager

from .file_handler import open_fsspec


def is_local_file(path: str) -> bool:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    return fs.protocol in ("file", None)


class PatchedAnnDataFileManager(OriginalAnnDataFileManager):
    """Backing file manager for AnnData."""

    def __repr__(self) -> str:
        if self.filename is None:
            return "(patched) Backing file manager: no file is set."
        else:
            return f"(patched) Backing file manager of file {self.filename}."

    @property
    def filename(self) -> Path:
        return self._filename

    @filename.setter
    def filename(self, filename: PathLike[str] | str | None):
        if filename is None:
            self._filename = None
            return
        self._filename = Path(filename) if is_local_file(filename) else filename

    def open(
        self,
        filename: PathLike[str] | str | None = None,
        filemode: Literal["r", "r+"] | None = None,
    ):
        if filename is not None:
            self.filename = filename
        if filemode is not None:
            self._filemode = filemode
        if self.filename is None:
            msg = "Cannot open backing file if backing not initialized."
            raise ValueError(msg)
        # Force open in read mode
        if is_local_file(self.filename):
            self._file = h5py.File(self.filename, "r")
        else:
            self._file = open_fsspec(self.filename)

    def close(self):
        """Close the backing file, remember filename, do *not* change to memory mode."""
        if self._file is not None:
            self._file.close()
        if self._fsspec_file is not None:
            self._fsspec_file.close()

    def _to_memory_mode(self):
        """Close the backing file, forget filename, *do* change to memory mode."""
        self._adata._X = self._adata.X[()]
        if self._fsspec_file is not None:
            self._fsspec_file.close()
        self._file.close()
        self._file = None
        self._filename = None


def apply_file_backing_patch():
    try:
        import anndata._core.anndata as anndata_module
        import anndata._core.file_backing as file_backing_module

        # patch the module where it is defined
        file_backing_module.AnnDataFileManager = PatchedAnnDataFileManager
        # patch the module where it is used
        anndata_module.AnnDataFileManager = PatchedAnnDataFileManager
        print("✓ Applied AnnDataFileManager patch, AnnData cannot be imported after the patch!")
    except ImportError:
        print("⚠ AnnDataFileManager not available for patching")


def rollback_file_backing_patch():
    import anndata._core.anndata as anndata_module
    import anndata._core.file_backing as file_backing_module

    file_backing_module.AnnDataFileManager = OriginalAnnDataFileManager
    anndata_module.AnnDataFileManager = OriginalAnnDataFileManager
