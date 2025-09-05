from urllib.parse import urlparse

import fsspec
import h5py


def open_fsspec(filename: str):
    parsed = urlparse(filename)
    scheme = parsed.scheme.lower()

    if scheme == "dnanexus":
        fs = fsspec.filesystem("dnanexus")
        file = fs.open(filename, mode="rb")
    else:
        # For local files or other supported fsspec schemes
        fs, path = fsspec.core.url_to_fs(filename)
        file = fs.open(path, mode="rb")

    return h5py.File(file, "r")
