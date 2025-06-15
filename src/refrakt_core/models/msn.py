"""
Momentum Self-Supervised Learning with Masked Siamese Networks (MSN).

This module implements the MSN training model with a backbone encoder and
projector + prototypes for clustering and contrastive learning.
"""

import copy
from typing import Tuple

import torch
import torch.nn.functional as F
from timm import create_model
from torch import Tensor, nn

from refrakt_core.models.templates.base import BaseModel
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.model_registry import register_model

@register_model("msn")
class MSNModel(BaseModel):
    """
    Masked Siamese Network (MSN) for self-supervised learning.

    Args:
        encoder_name (str): Name of the encoder backbone (from timm).
        projector_dim (int): Output dimension of the projector.
        num_prototypes (int): Number of prototypes for clustering.
        pretrained (bool): Whether to load pretrained encoder weights.
    """

    def __init__(
        self,
        encoder_name: str,
        projector_dim: int,
        num_prototypes: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Online and target encoders
        self.encoder: nn.Module = create_model(
            encoder_name, pretrained=pretrained, num_classes=0
        )
        self.target_encoder: nn.Module = create_model(
            encoder_name, pretrained=False, num_classes=0
        )

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        dim = projector_dim
        self.projector: nn.Module = nn.Sequential(
            nn.BatchNorm1d(self.encoder.num_features),
            nn.Linear(self.encoder.num_features, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, affine=False),
        )

        self.target_projector: nn.Module = copy.deepcopy(self.projector)
        for param in self.target_projector.parameters():
            param.requires_grad = False

        self.prototypes: nn.Parameter = nn.Parameter(torch.randn(num_prototypes, dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x_anchor: Tensor, x_target: Tensor) -> ModelOutput:
        """
        Forward pass for MSN training.
        Returns a ModelOutput with embeddings and auxiliary data.
        """
        z_anchor: Tensor = self.encoder(x_anchor)
        z_anchor = self.projector(z_anchor)
        z_anchor = F.normalize(z_anchor, dim=-1)

        with torch.no_grad():
            z_target: Tensor = self.target_encoder(x_target)
            z_target = self.target_projector(z_target)
            z_target = F.normalize(z_target, dim=-1)

        return ModelOutput(
            embeddings=z_anchor,
            loss_components={
                "z_target": z_target,
                "prototypes": self.prototypes
            }
        )


    def __call__(self, *args, **kwargs):
        """
        Override to handle W&B or TorchScript tracing (which pass a single tensor).
        If only one input is passed, we assume it's a dummy call and replicate it.
        """
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            dummy = args[0]
            return self.forward(dummy, dummy)
        return super().__call__(*args, **kwargs)

    def dummy_forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.forward(x, x)