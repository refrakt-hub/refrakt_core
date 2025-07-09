"""
Decoder components for transformer-style architectures.

Modules:
- DecoderBlock
- Decoder
"""

from typing import Optional, cast

from torch import Tensor, nn

from refrakt_core.utils.classes.attention import MHA
from refrakt_core.utils.classes.resnet import SkipConnections
from refrakt_core.utils.classes.utils import FeedForward, LayerNormalization


class DecoderBlock(nn.Module):
    """
    A single transformer decoder block with masked self-attention,
    cross-attention, and feedforward layers, each wrapped with residual connections.

    Args:
        masked_att (MHA): Multi-head masked self-attention module.
        cross_att (MHA): Multi-head cross-attention module.
        feed_forw (FeedForward): Feedforward network.
        dropout (float): Dropout rate.
    """

    def __init__(
        self, masked_att: MHA, cross_att: MHA, feed_forw: FeedForward, dropout: float
    ) -> None:
        super().__init__()
        self.masked_att = masked_att
        self.cross_att = cross_att
        self.feed_forw = feed_forw
        self.dropout = nn.Dropout(dropout)
        self.skip_conn = nn.ModuleList([SkipConnections(dropout) for _ in range(3)])

    def forward(
        self,
        x: Tensor,
        enc_output: Tensor,
        src_mask: Optional[Tensor],
        tgt_mask: Optional[Tensor],
    ) -> Tensor:
        """
        Forward pass through the decoder block.

        Args:
            x (Tensor): Target sequence input of shape (B, T, D).
            enc_output (Tensor): Encoder output of shape (B, S, D).
            src_mask (Optional[Tensor]): Source attention mask.
            tgt_mask (Optional[Tensor]): Target attention mask.

        Returns:
            Tensor: Output after decoder block processing.
        """
        x = self.skip_conn[0](x, lambda x: self.masked_att(x, x, x, tgt_mask))
        x = self.skip_conn[1](
            x, lambda x: self.cross_att(x, enc_output, enc_output, src_mask)
        )
        x = self.skip_conn[2](x, self.feed_forw)
        return x  # type: ignore[no-any-return]


class Decoder(nn.Module):
    """
    Transformer decoder consisting of multiple decoder blocks.

    Args:
        layers (nn.ModuleList): List of DecoderBlock modules.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(
        self,
        x: Tensor,
        enc_output: Tensor,
        src_mask: Optional[Tensor],
        tgt_mask: Optional[Tensor],
    ) -> Tensor:
        """
        Forward pass through all decoder layers.

        Args:
            x (Tensor): Target input tensor of shape (B, T, D).
            enc_output (Tensor): Encoder output tensor of shape (B, S, D).
            src_mask (Optional[Tensor]): Source attention mask.
            tgt_mask (Optional[Tensor]): Target attention mask.

        Returns:
            Tensor: Final normalized decoder output.
        """
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return cast(Tensor, self.norm(x))
