"""
Common utility classes used throughout the Refrakt core repository.

Modules:
- LayerNormalization: Custom layer normalization layer.
- MLPHead: Multi-layer perceptron head with weight init.
- Merge: Rearranges and merges image patches.
- Projection: Applies a projection to model output.
- FeedForward: Standard FFN block used in transformers.
"""

import math
from typing import cast

import torch
from einops import rearrange
from torch import Tensor, nn


class LayerNormalization(nn.Module):
    """
    Custom implementation of layer normalization using learnable scale and bias.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha: nn.Parameter = nn.Parameter(torch.ones(1))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(1))
        self.eps: float = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies layer normalization.

        Args:
            x (Tensor): Input tensor of shape (batch_size, ..., features)

        Returns:
            Tensor: Normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class MLPHead(nn.Module):
    """
    MLP Head with three linear layers and GELU activations.

    Args:
        in_dim (int): Input dimension.
        hidden_dim (int): Hidden layer dimension. Defaults to 2048.
        out_dim (int): Output dimension. Defaults to 65536.
    """

    def __init__(
        self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 65536
    ) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes weights with truncated normal and biases to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through MLP head.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return cast(Tensor, self.net(x))


class Merge(nn.Module):
    """
    Merges 2x2 spatial patches and applies linear projection + layer normalization.

    Args:
        C (int): Input channel dimension.
    """

    def __init__(self, C: int) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(4 * C, 2 * C)
        self.norm: nn.LayerNorm = nn.LayerNorm(2 * C)

    def forward(self, x: Tensor) -> Tensor:
        """
        Merges patches and projects.

        Args:
            x (Tensor): Input tensor of shape (B, H*W, C)

        Returns:
            Tensor: Transformed tensor of shape (B, H/2 * W/2, 2C)
        """
        height = width = int(math.sqrt(x.shape[1]) / 2)
        x = rearrange(
            x, "b (h s1 w s2) c -> b (h w) (s2 s1 c)", s1=2, s2=2, h=height, w=width
        )
        x = self.linear(x)
        x = self.norm(x)
        return cast(Tensor, x)


class Projection(nn.Module):
    """
    Applies a linear projection followed by log softmax.

    Args:
        d_model (int): Model dimension.
        vocab_size (int): Vocabulary size.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj: nn.Linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Projects input to vocab size with log softmax.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor: Log-softmax over vocabulary.
        """
        return torch.log_softmax(self.proj(x), dim=-1)


class FeedForward(nn.Module):
    """
    Standard feedforward network used in transformer blocks.

    Args:
        d_model (int): Model dimension.
        d_ff (int): Hidden layer dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1: nn.Linear = nn.Linear(d_model, d_ff)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.linear_2: nn.Linear = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies feedforward transformation with ReLU and dropout.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Transformed tensor.
        """
        x = self.linear_1(x)
        x = self.dropout(torch.relu(x))
        x = self.linear_2(x)
        return cast(Tensor, x)
