import os
import time

# Apply daft patches for per-CPU workers
import anndata
import daft

from protoplast.patches.anndata_read_h5ad_backed import apply_read_h5ad_backed_patch
from protoplast.patches.anndata_remote import apply_file_backing_patch
from protoplast.patches.daft_flotilla import apply_flotilla_patches

os.environ["MAX_WORKERS"] = "2"
apply_flotilla_patches()
apply_file_backing_patch()
apply_read_h5ad_backed_patch()

daft.context.set_execution_config(native_parquet_writer=False)
file_path = "s3://anyscale-ap-data/test_medium.h5ad"


def test_read_h5ad():
    start = time.time()
    ad = anndata.read_h5ad(file_path, backed="r")
    print(ad.X.shape)
    print(ad.X[1, :])
    print(f"Time took: {time.time() - start}")


if __name__ == "__main__":
    test_read_h5ad()
