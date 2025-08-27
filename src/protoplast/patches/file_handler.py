import fsspec
import h5py


def open_fsspec(filename: str):
    if "dnanexus" in filename:
        dxfs = fsspec.filesystem("dnanexus")
        file = dxfs.open(filename, mode="rb")
        return h5py.File(file, "r")
    else:
        file = fsspec.open(filename, mode="rb")
        return h5py.File(file.open(), "r")
