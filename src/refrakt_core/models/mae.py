from typing import Dict

import torch
from torch import Tensor, nn
from einops import rearrange

from refrakt_core.models.templates.base import BaseModel
from refrakt_core.registry.model_registry import register_model
from refrakt_core.utils.methods import get_2d_sincos_pos_embed, random_masking
from refrakt_core.schema.model_output import ModelOutput


@register_model("mae")
class MAE(BaseModel):
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
        super().__init__({})
        self.mask_ratio = mask_ratio

        self.num_patches: int = (img_size // patch_size) ** 2
        self.patch_dim: int = patch_size * patch_size * in_chans

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed_enc = nn.Parameter(
            get_2d_sincos_pos_embed(embed_dim, int(self.num_patches**0.5), cls_token=False),
            requires_grad=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(decoder_dim, int(self.num_patches**0.5), cls_token=False),
            requires_grad=False,
        )
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        decoder_layer = nn.TransformerEncoderLayer(
            decoder_dim, decoder_num_heads, dim_feedforward=decoder_dim * 4, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.constant_(self.decoder_pred.bias, 0)

    def patchify(self, imgs: Tensor) -> Tensor:
        p = self.patch_embed.kernel_size[0]
        return rearrange(imgs, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)

    def unpatchify(self, patches: Tensor) -> Tensor:
        p = self.patch_embed.kernel_size[0]
        h = w = int(patches.shape[1] ** 0.5)
        return rearrange(
            patches, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=p, p2=p, c=3
        )

    def forward(self, imgs: Tensor) -> ModelOutput:
        # Patch embedding + positional encoding
        x = self.patch_embed(imgs)                       # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)                 # [B, N, embed_dim]
        x = x + self.pos_embed_enc.unsqueeze(0)         

        # Masking and encoding
        x_masked, mask, ids_restore, _ = random_masking(x, self.mask_ratio)
        encoded = self.encoder(x_masked)

        # Decoder
        decoded_tokens = self.decoder_embed(encoded)
        B, N_visible, C = decoded_tokens.shape
        N_masked = self.num_patches - N_visible

        mask_tokens = self.mask_token.expand(B, N_masked, -1)
        full_tokens = torch.zeros(B, self.num_patches, C, device=imgs.device)
        full_tokens.scatter_(
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, C),
            torch.cat([decoded_tokens, mask_tokens], dim=1),
        )

        full_tokens = full_tokens + self.decoder_pos_embed.unsqueeze(0)
        decoded = self.decoder(full_tokens)
        pred = self.decoder_pred(decoded)                # [B, num_patches, patch_dim]

        # Ground truth
        original_patches = self.patchify(imgs)

        return ModelOutput(
            reconstruction=pred,
            extra={
                "mask": mask,                             # [B, num_patches]
                "original_patches": original_patches,    # [B, num_patches, patch_dim]
                "ids_restore": ids_restore,              # [B, num_patches]
                "decoded_tokens": decoded,               # optional: [B, num_patches, decoder_dim]
            },
        )
