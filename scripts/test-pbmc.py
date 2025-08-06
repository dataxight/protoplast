from protoplast.genomics.anndata import read_h5ad
import daft
import time
from daft.logging import setup_debug_logger

setup_debug_logger()
def main():
    daft.context.set_runner_ray()
    daft.context.set_execution_config(scan_tasks_min_size_bytes=100000)
    df = read_h5ad("/home/tphan/pbmc_seurat_v4.h5ad", batch_size=5000)
    print(df.count().show())

if __name__ == "__main__":
    main()