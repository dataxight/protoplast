from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Literal

import fsspec
import h5py
from anndata._core.file_backing import AnnDataFileManager as OriginalAnnDataFileManager

from protoplast.utils import REMOTE_FILE_DRIVER, get_remote_file_object, is_local_file


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
            self._fsspec_file = get_remote_file_object(self.filename, driver=REMOTE_FILE_DRIVER)
            self._file = h5py.File(self._fsspec_file, "r")

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
        print("✓ Applied AnnDataFileManager patch")
    except ImportError:
        print("⚠ AnnDataFileManager not available for patching")


def rollback_file_backing_patch():
    import anndata._core.anndata as anndata_module
    import anndata._core.file_backing as file_backing_module

    file_backing_module.AnnDataFileManager = OriginalAnnDataFileManager
    anndata_module.AnnDataFileManager = OriginalAnnDataFileManager
