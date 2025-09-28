#   Copyright 2025 DataXight, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import time

import anndata as ad
import numpy as np
import torch
from anndata.experimental import AnnCollection
from lightning.pytorch import Trainer
from scdataset import BlockShuffling, Streaming, scDataset
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from protoplast.scrna.anndata.lightning_models import LinearClassifier


def train(
    paths: str, batch_size: int = 64, num_workers: int = 12, fetch_factor: int = 16, is_shuffle=False, block_size=4
):
    start = time.time()
    adatas = [ad.read_h5ad(f, backed="r") for f in paths if f.endswith("h5ad")]
    for data in adatas:
        data.obs = data.obs[["cell_line"]]

    collection = AnnCollection(adatas)
    cell_line_encoder = LabelEncoder()
    cell_line_encoder.fit(collection.obs["cell_line"].values)
    print(f"Data loading time: {time.time() - start:.2f} seconds")

    def fetch_transform(batch):
        # Materialize the AnnData batch (X matrix) in memory
        return batch.to_adata()

    def batch_transform(batch, cell_line_encoder=cell_line_encoder):
        # Convert AnnData batch to (X, y) tensors for training
        X = batch.X.astype("float32")
        # Densify if X is a sparse matrix
        if sparse.issparse(X):
            X = X.toarray()
        X = torch.from_numpy(X)
        y = cell_line_encoder.transform(batch.obs["cell_line"].values)
        y = torch.from_numpy(y).long()
        return X, y

    indices = np.arange(collection.n_obs)

    start = time.time()
    # Stratified train/test split on cell_line
    print("Splitting data into train and test sets")
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    print(f"Data splitting time: {time.time() - start:.2f} seconds")

    if is_shuffle:
        train_strategy = BlockShuffling(indices=train_idx, block_size=block_size)
    else:
        train_strategy = Streaming(indices=train_idx)
    val_strategy = Streaming(indices=test_idx)
    scdata_train = scDataset(
        data_collection=collection,
        strategy=train_strategy,
        batch_size=batch_size,
        fetch_factor=fetch_factor,
        fetch_transform=fetch_transform,
        batch_transform=batch_transform,
    )
    scdata_test = scDataset(
        data_collection=collection,
        strategy=val_strategy,
        batch_size=batch_size,
        fetch_factor=fetch_factor,
        fetch_transform=fetch_transform,
        batch_transform=batch_transform,
    )

    train_loader = DataLoader(
        scdata_train,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=fetch_factor + 1,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        scdata_test,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=fetch_factor + 1,
        persistent_workers=True,
        pin_memory=True,
    )

    model = LinearClassifier(num_genes=collection.shape[1], num_classes=len(cell_line_encoder.classes_))
    trainer = Trainer(max_epochs=1)
    print("Starting training")
    trainer.fit(model, train_loader, val_loader)
    print(trainer.callback_metrics)
