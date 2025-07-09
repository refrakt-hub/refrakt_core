"""Encoder block and full encoder module for transformer architectures."""

from typing import cast

from torch import Tensor, nn

from refrakt_core.utils.classes.attention import MHA
from refrakt_core.utils.classes.resnet import SkipConnections
from refrakt_core.utils.classes.utils import FeedForward, LayerNormalization


class EncoderBlock(nn.Module):
    """
    A single encoder block composed of multi-head self-attention,
    feed-forward network, and skip connections with dropout.
    """

    def __init__(self, self_att: MHA, feed_forw: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_att = self_att
        self.feed_forw = feed_forw
        self.dropout = nn.Dropout(dropout)
        self.skip_conn = nn.ModuleList([SkipConnections(dropout) for _ in range(2)])

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        """
        Forward pass through the encoder block.

        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, embed_dim].
            src_mask (Tensor): Source mask for attention.

        Returns:
            Tensor: Output tensor after attention and feedforward layers.
        """
        x = self.skip_conn[0](x, lambda y: self.self_att(y, y, y, src_mask))
        x = self.skip_conn[1](x, self.feed_forw)
        return x  # type: ignore[no-any-return]


class Encoder(nn.Module):
    """
    Stacked transformer encoder composed of multiple EncoderBlocks.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, embed_dim].
            mask (Tensor): Source mask for attention.

        Returns:
            Tensor: Output tensor after all encoder layers.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return cast(Tensor, self.norm(x))
