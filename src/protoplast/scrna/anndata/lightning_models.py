import lightning.pytorch as pl
import torch
from torch import nn


class LinearClassifier(pl.LightningModule):
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
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
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


class NullClassifier(pl.LightningModule):
    """
    Null model baseline: ignores input features, outputs uniform logits or a learnable bias.
    """

    def __init__(self, num_classes, learn_bias: bool = True):
        super().__init__()
        if learn_bias:
            # Learnable bias for each class (like always predicting priors)
            self.bias = nn.Parameter(torch.zeros(num_classes))
        else:
            # Fixed uniform distribution over classes
            self.register_buffer("bias", torch.zeros(num_classes))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Ignore x entirely, just return the bias repeated for each sample
        batch_size = x.shape[0]
        logits = self.bias.repeat(batch_size, 1)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch  # need labels here
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # If bias is fixed (learn_bias=False), no optimizer needed
        if len(list(self.parameters())) == 0:
            return []
        return torch.optim.Adam(self.parameters(), lr=1e-2)
