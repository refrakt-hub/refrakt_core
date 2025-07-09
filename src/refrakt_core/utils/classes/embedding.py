"""
Embedding modules used in vision transformer architectures.

Modules:
- Embedding
- RelativeEmbedding
"""

import torch
from einops import rearrange
from torch import Tensor, nn


class Embedding(nn.Module):
    """
    Patch embedding module that projects image patches into a higher-dimensional space.

    Args:
        patch_size (int): The size of each image patch (default: 4).
        C (int): Output embedding dimension (default: 96).
        in_channels (int): Number of input channels in the image (default: 3).
    """

    def __init__(self, patch_size: int = 4, C: int = 96, in_channels: int = 3) -> None:
        super().__init__()
        self.linear: nn.Conv2d = nn.Conv2d(
            in_channels, C, kernel_size=patch_size, stride=patch_size
        )
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(C)
        self.relu: nn.ReLU = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for patch embedding.

        Args:
            x (Tensor): Input tensor of shape (B, C_in, H, W)

        Returns:
            Tensor: Patch-embedded tensor of shape (B, N_patches, C)
        """
        x = self.linear(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.relu(self.layer_norm(x))
        return x  # type: ignore[no-any-return]


class RelativeEmbedding(nn.Module):
    """
    Relative positional embedding module for 2D spatial positions.

    Args:
        window_size (int): Size of the attention window (default: 7).
    """

    def __init__(self, window_size: int = 7) -> None:
        super().__init__()

        # Declare buffer size
        self.window_size = window_size
        param = torch.randn(2 * window_size - 1, 2 * window_size - 1)
        self.register_parameter("base_param", nn.Parameter(param))

        x = torch.arange(1, window_size + 1, 1 / window_size)
        x_diff = (x[None, :] - x[:, None]).int()
        y_raw = torch.cat([torch.arange(1, window_size + 1)] * window_size)
        y_diff = y_raw[None, :] - y_raw[:, None]

        embedding_values = param[x_diff[:, :], y_diff[:, :]].detach()
        self.embeddings: nn.Parameter = nn.Parameter(
            embedding_values, requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies relative positional embedding to input tensor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Tensor with relative positional embeddings added.
        """
        return x + self.embeddings
