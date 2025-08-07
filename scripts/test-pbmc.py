from protoplast.genomics.anndata import read_h5ad
import daft
import time
from daft.logging import setup_debug_logger

# Apply daft patches for per-CPU workers
from protoplast.patches.daft_flotilla import apply_flotilla_patches
apply_flotilla_patches()

def main():
    daft.context.set_runner_ray()
    daft.context.set_execution_config(scan_tasks_min_size_bytes=100000)
    df = read_h5ad("/Users/tanphan/Downloads/pbmc_seurat_v4.h5ad", batch_size=5000)
    start = time.time()
    print(df.select("TP53").count().show())
    print(f"Time took: {time.time() - start}")

if __name__ == "__main__":
    main()