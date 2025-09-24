from protoplast.scrna.anndata.gds import save_to_gds
from protoplast.scrna.anndata.strategy import SequentialShuffleStrategy
from protoplast.scrna.anndata.torch_dataloader import DistributedCellLineAnnDataset, cell_line_metadata_cb

if __name__ == "__main__":
    save_to_gds(
        [f"/ephemeral/tahoe100m/plate{i+1}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad" for i in range(13)],
        SequentialShuffleStrategy,
        DistributedCellLineAnnDataset,
        "/ephemeral/gds/all_plates",
        metadata_cb=cell_line_metadata_cb,
    )
