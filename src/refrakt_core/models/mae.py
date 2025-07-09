"""
Masked Autoencoders (MAE) for self-supervised learning.

This module implements the core MAE architecture for image reconstruction from
randomly masked patches using a vision transformer encoder-decoder setup.
"""

import torch
from einops import rearrange
from torch import Tensor, nn

from refrakt_core.models.templates.base import BaseModel
from refrakt_core.registry.model_registry import register_model
from refrakt_core.utils.methods import get_2d_sincos_pos_embed, random_masking


@register_model("mae")
class MAE(BaseModel):
    """
    Masked Autoencoder (MAE) implementation for image reconstruction.

    Args:
        img_size (int): Size of input image (assumed square).
        patch_size (int): Size of patch.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of patch embeddings.
        encoder_depth (int): Depth of encoder transformer.
        decoder_dim (int): Dimension of decoder embeddings.
        decoder_depth (int): Depth of decoder transformer.
        num_heads (int): Number of encoder attention heads.
        decoder_num_heads (int): Number of decoder attention heads.
        mask_ratio (float): Fraction of patches to mask.
    """

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        num_heads: int = 12,
        decoder_num_heads: int = 16,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

        self.num_patches: int = (img_size // patch_size) ** 2
        self.patch_dim: int = patch_size * patch_size * in_chans

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed_enc = nn.Parameter(
            get_2d_sincos_pos_embed(
                embed_dim, int(self.num_patches**0.5), cls_token=False
            ),
            requires_grad=False,
        )

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Decoder
        self.decoder_pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(
                decoder_dim, int(self.num_patches**0.5), cls_token=False
            ),
            requires_grad=False,
        )
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        decoder_layer = nn.TransformerEncoderLayer(
            decoder_dim,
            decoder_num_heads,
            dim_feedforward=decoder_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Initialize weights for decoder and mask token.
        """
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.constant_(self.decoder_pred.bias, 0)

    def patchify(self, imgs: Tensor) -> Tensor:
        """
        Split images into patches.

        Args:
            imgs (Tensor): Shape (B, C, H, W)

        Returns:
            Tensor: Shape (B, N, patch_dim)
        """
        p = self.patch_embed.kernel_size[0]
        return rearrange(imgs, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)

    def unpatchify(self, patches: Tensor) -> Tensor:
        """
        Reconstruct image from patches.

        Args:
            patches (Tensor): Shape (B, N, patch_dim)

        Returns:
            Tensor: Shape (B, C, H, W)
        """
        p = self.patch_embed.kernel_size[0]
        h = w = int(patches.shape[1] ** 0.5)
        return rearrange(
            patches, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=p, p2=p, c=3
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the MAE model.

        Args:
            x (Tensor): Input images of shape (B, 3, H, W)

        Returns:
            dict: {
                'recon_patches': Reconstructed patch pixels,
                'mask': Mask binary array,
                'original_patches': Original unmasked patch tokens
            }
        """
        # === Patch embedding ===
        x_patch = self.patch_embed(x)  # [B, C, H', W']
        x_patch = x_patch.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x_patch = x_patch + self.pos_embed_enc.unsqueeze(0)

        # === Random masking ===
        x_masked, mask, ids_restore, _ = random_masking(x_patch, self.mask_ratio)

        # === Encoder ===
        encoded = self.encoder(x_masked)

        # === Decoder input assembly ===
        decoded_tokens = self.decoder_embed(encoded)
        batch_size, num_visible, channels = decoded_tokens.shape
        num_masked = self.num_patches - num_visible

        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)

        full_tokens = torch.zeros(
            batch_size, self.num_patches, channels, device=x.device
        )
        full_tokens.scatter_(
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, channels),
            torch.cat([decoded_tokens, mask_tokens], dim=1),
        )

        full_tokens = full_tokens + self.decoder_pos_embed.unsqueeze(0)
        decoded = self.decoder(full_tokens)

        # === Prediction ===
        pred = self.decoder_pred(decoded)

        return {
            "recon": pred,
            "mask": mask,
            "original_patches": self.patchify(x),
        }  # type: ignore[return-value]
