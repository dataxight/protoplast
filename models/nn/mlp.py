"""
Multi-layer perceptron and mask network implementations.
"""


import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable layers and dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        dropout: float | None = None,
        activation: str = "gelu"
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self._get_activation(activation))
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer (only if we have more than 1 layer)
        if n_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            # Single layer case
            layers = [nn.Linear(input_dim, output_dim)]

        self.net = nn.Sequential(*layers)

    def _get_activation(self, activation: str):
        """Get activation function by name."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.net(x)


class MaskNet(nn.Module):
    """
    Mask network for sparse additive mechanism.
    Produces attention-like masks for perturbation effects.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        dropout: float | None = None
    ):
        super().__init__()

        # Use MLP backbone
        self.backbone = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout,
            activation="relu"
        )

        # Final sigmoid to ensure mask values are in [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass producing mask values.
        
        Args:
            x: Input tensor
            
        Returns:
            Mask tensor with values in [0, 1]
        """
        logits = self.backbone(x)
        return self.sigmoid(logits)


class ResidualMLP(nn.Module):
    """
    MLP with residual connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # Projection layer for residual connection if dimensions don't match
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.proj(x) + self.net(x)
