import torch
import torch.nn as nn
import numpy as np

def to_dense(batch_x):
    if hasattr(batch_x, "toarray"):  # sparse
        return torch.tensor(batch_x.toarray(), dtype=torch.float32)
    if isinstance(batch_x, np.ndarray):
        return torch.tensor(batch_x, dtype=torch.float32)
    return batch_x.float()

def make_token_batch(X_dense, k=256):
    # X_dense: (B, G)
    B, G = X_dense.shape
    # top-k per row
    topk_vals, topk_idx = torch.topk(X_dense, k=min(k, G), dim=1)  # (B, k)
    return topk_idx.long(), topk_vals.float()

# We'll wrap original loaders with a generator that transforms batches on the fly
def token_loader(ann_loader, topk_tokens=256):
    for b in ann_loader:
        X = to_dense(b["X"])             # (B, G)
        yi = torch.tensor(b["y"].astype(int)) if isinstance(b["y"], np.ndarray) else b["y"]
        tok_ids, tok_vals = make_token_batch(X, k=topk_tokens)
        yield {"tok_ids": tok_ids, "tok_vals": tok_vals, "y": yi}

# ---------------------------
# 4) Transformer model
# ---------------------------
class ExprTransformer(nn.Module):
    def __init__(self, num_genes, num_classes, d_model=128, n_heads=4, n_layers=4, dropout=0.1, topk_tokens=256):
        super().__init__()
        self.gene_emb = nn.Embedding(num_genes, d_model)
        # scalar expression -> embedding
        self.val_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + topk_tokens, d_model))  # CLS + tokens
        self.topk_tokens = topk_tokens

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.gene_emb.weight, std=0.02)

    def forward(self, tok_ids, tok_vals):
        """
        tok_ids: (B, K) int64
        tok_vals: (B, K) float32
        """
        B, K = tok_ids.shape
        g = self.gene_emb(tok_ids)                             # (B, K, D)
        v = self.val_mlp(tok_vals.unsqueeze(-1))               # (B, K, D)
        x = g + v                                              # (B, K, D)

        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                         # (B, 1+K, D)
        x = x + self.pos_emb[:, : (1+K), :]                    # add positional emb (optional)

        h = self.encoder(x)                                    # (B, 1+K, D)
        h_cls = self.norm(h[:, 0])                             # (B, D)
        logits = self.head(h_cls)                              # (B, C)
        return logits

    @torch.no_grad()
    def encode(self, tok_ids, tok_vals):
        """Return the CLS embedding for each cell without applying the head.

        Args:
            tok_ids: Tensor of shape (B, K) with gene token indices
            tok_vals: Tensor of shape (B, K) with expression values
        Returns:
            Tensor of shape (B, D) representing per-cell embeddings.
        """
        self.eval()
        B, K = tok_ids.shape
        g = self.gene_emb(tok_ids)
        v = self.val_mlp(tok_vals.unsqueeze(-1))
        x = g + v

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb[:, : (1+K), :]

        h = self.encoder(x)
        h_cls = self.norm(h[:, 0])
        return h_cls