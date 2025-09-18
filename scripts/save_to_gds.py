from protoplast.scrna.anndata.gds import save_to_gds
from protoplast.scrna.anndata.strategy import SequentialShuffleStrategy
from protoplast.scrna.anndata.torch_dataloader import cell_line_metadata_cb, DistributedCellLineAnnDataset

    

if __name__ == "__main__":
    save_to_gds(
        ["/mnt/ham/dx_data/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"],
        SequentialShuffleStrategy,
        DistributedCellLineAnnDataset,
        "/mnt/ham/dx_data/plate3_gpu",
        metadata_cb=cell_line_metadata_cb
    )

