# SCRNA API References

## Trainer

:::protoplast.scrna.anndata.trainer.RayTrainRunner

## DataModule
Wrapper around Dataset on how the data should be forward to the Lightning Model support hooks at various
Lifecycle when the data get pass to the model

:::protoplast.scrna.anndata.torch_dataloader.AnnDataModule


## Dataset
For fetching and sending data to the model
:::protoplast.scrna.anndata.torch_dataloader.DistributedAnnDataset


:::protoplast.scrna.anndata.strategy.ShuffleStrategy

:::protoplast.scrna.anndata.strategy.SplitInfo

:::protoplast.scrna.anndata.strategy.SequentialShuffleStrategy
