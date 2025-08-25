import lightning.pytorch as pl
from torch import nn
import torch


class BaseAnnDataLightningModule(pl.LightningModule):
    def densify(self, x: torch.Tensor):
        """
        Densify a sparse matrix in GPU how you use
        this depends on what your Dataset yield for each batch
        consult the documentation for more information
        """
        if x.is_sparse or x.is_sparse_csr:
            x = x.to_dense()
        return x


class LinearClassifier(BaseAnnDataLightningModule):
    """
    Example model for implementing the cell line linear classifier
    you can write your own model by extending BaseAnnDataLightningModule
    it is highly recommend to extend from this class if you are using
    the DistributedAnnDataset as your loader
    """
    def __init__(self, num_genes, num_classes):
        super().__init__()
        self.model = nn.Linear(num_genes, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.densify(x)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.densify(x)
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        acc = correct / total
        self.log("val_acc", acc)
        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer