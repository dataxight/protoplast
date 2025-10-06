"""
scVI-like encoder/decoder blocks implemented with the local MLP utility.

This file provides:
- Encoder: parameterizes q(z|x) with Normal(mean, var) and reparameterized sample
- DecoderSCVI: maps z (and optional covariates) to NB/ZINB parameters

We use the local MLP instead of scvi-tools FCLayers for minimal dependency.
"""

from typing import Callable, Iterable, Literal

import torch
import torch.nn as nn
from torch.distributions import Normal

from .mlp import MLP


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


class Encoder(nn.Module):
    """Encode data of ``n_input`` dims into latent of ``n_output`` dims.

    The latent is modeled with a diagonal Normal. We expose both the parameters
    and a reparameterized sample.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        return_dist: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        # Use local MLP to mimic scvi FCLayers
        self.encoder = MLP(
            input_dim=n_input,
            hidden_dim=n_hidden,
            output_dim=n_hidden,
            n_layers=max(1, n_layers),
            dropout=dropout_rate,
            activation="gelu",
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):
        # Parameters for latent distribution
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent


class DecoderSCVI(nn.Module):
    """Decode latent ``z`` to parameters of a ZINB distribution.

    We return:
    - px_scale: gene-wise mean proportions (softmax or softplus)
    - px_r: gene-wise dispersion (unconstrained; apply softplus later)
    - px_rate: expected count mean given library size (exp(library) * px_scale)
    - px_dropout: logits for zero-inflation probability
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        **kwargs,
    ):
        super().__init__()

        self.px_decoder = MLP(
            input_dim=n_input,
            hidden_dim=n_hidden,
            output_dim=n_hidden,
            n_layers=max(1, n_layers),
            dropout=0.0,
            activation="gelu",
        )

        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        else:
            raise ValueError("Unsupported scale_activation")

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )

        # dispersion (gene-cell)
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout logits
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
    ):
        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


