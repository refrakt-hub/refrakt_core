"""
Class implementations that cover the critical blocks of the Swin Transformer.

Modules:
- SwinBlock
- AlternateBlock
"""

from typing import cast

from torch import Tensor, nn

from refrakt_core.utils.classes.attention import ShiftedWindowMSA


class SwinBlock(nn.Module):
    """
    A Swin Transformer block containing LayerNorm, (Shifted) Window MSA, and MLP.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        window_size (int): Size of attention windows.
        mask (bool): Whether to apply masking (for shifted windows).
    """

    def __init__(self, embed_dim: int, num_heads: int, window_size: int, mask: bool):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.wmsa = ShiftedWindowMSA(
            embed_dim=embed_dim, n_heads=num_heads, window_size=window_size, mask=mask
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the SwinBlock.

        Args:
            x (Tensor): Input tensor of shape (B, H*W, C).

        Returns:
            Tensor: Output tensor of shape (B, H*W, C).
        """
        res1 = self.dropout(self.wmsa(self.layer_norm(x)) + x)
        x = self.layer_norm(res1)
        x = self.mlp(x)
        return cast(Tensor, self.dropout(x + res1))


class AlternateSwin(nn.Module):
    """
    A module containing two SwinBlocks with alternating attention patterns:
    one with regular WSA and one with shifted WMSA.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        window_size (int): Size of attention windows. Default is 7.
    """

    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 7):
        super().__init__()
        self.wsa = SwinBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            mask=False,
        )
        self.wmsa = SwinBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            mask=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through alternating WSA and WMSA blocks.

        Args:
            x (Tensor): Input tensor of shape (B, H*W, C).

        Returns:
            Tensor: Output tensor after two Swin blocks.
        """
        return cast(Tensor, self.wmsa(self.wsa(x)))
