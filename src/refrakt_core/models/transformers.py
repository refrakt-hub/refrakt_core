"""
Transformer architecture module for sequence-to-sequence tasks.

Combines embedding, positional encoding, encoder, and decoder blocks,
mirroring the original Transformer architecture.
"""

from torch import Tensor, nn

from refrakt_core.utils.classes.decoder import Decoder
from refrakt_core.utils.classes.encoder import Encoder
from refrakt_core.utils.classes.transformers import InputEmbeddings, PositionalEncoding
from refrakt_core.utils.classes.utils import Projection


class Transformer(nn.Module):
    """
    Transformer model composed of encoder, decoder, and projection layers.

    Args:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        src_embed (InputEmbeddings): Source token embedding.
        tgt_embed (InputEmbeddings): Target token embedding.
        src_pos (PositionalEncoding): Positional encoding for source.
        tgt_pos (PositionalEncoding): Positional encoding for target.
        proj (Projection): Projection layer from decoder to output space.
    """

    encoder: Encoder
    _decoder: Decoder
    src_embed: InputEmbeddings
    tgt_embed: InputEmbeddings
    src_pos: PositionalEncoding
    tgt_pos: PositionalEncoding
    proj: Projection

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        proj: Projection,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self._decoder = decoder  # renamed to avoid name collision
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Encode source input using embedding and encoder.

        Args:
            src: Source token IDs or embeddings.
            src_mask: Source attention mask.

        Returns:
            Encoder output tensor.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)  # type: ignore[no-any-return]

    def decode(
        self,
        enc_output: Tensor,
        src_mask: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """
        Decode target input using encoder output.

        Args:
            enc_output: Output from encoder.
            src_mask: Source attention mask.
            tgt: Target token IDs or embeddings.
            tgt_mask: Target attention mask.

        Returns:
            Decoder output tensor.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self._decoder(tgt, enc_output, src_mask, tgt_mask)  # type: ignore[no-any-return]

    def project(self, x: Tensor) -> Tensor:
        """
        Project decoder output to output vocabulary.

        Args:
            x: Decoder output.

        Returns:
            Logits for output vocabulary.
        """
        return self.proj(x)  # type: ignore[no-any-return]

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        """
        Full forward pass for the Transformer.

        Args:
            src: Source input.
            tgt: Target input.
            src_mask: Source attention mask.
            tgt_mask: Target attention mask.

        Returns:
            Projected logits for output sequence.
        """
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(enc_output, src_mask, tgt, tgt_mask)
        return self.project(dec_output)
