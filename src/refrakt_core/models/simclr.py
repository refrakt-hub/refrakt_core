"""
SimCLR contrastive learning model using a ResNet-50 backbone.

Includes projection head and custom training/validation step hooks for contrastive loss.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models  # type: ignore

from refrakt_core.models.templates.models import BaseContrastiveModel
from refrakt_core.registry.model_registry import register_model


@register_model("simclr")
class SimCLRModel(BaseContrastiveModel):
    """
    SimCLR Model implementation.

    Attributes:
        encoder (nn.Module): ResNet-50 backbone with the FC layer removed.
        projector (nn.Module): MLP projection head for contrastive representation learning.
    """

    def __init__(self, proj_dim: int = 128) -> None:
        super().__init__(
            model_name="simclr", backbone_name="resnet50", proj_dim=proj_dim
        )

        self.encoder: nn.Module = models.resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()  # type: ignore[assignment]

        self.projector: nn.Module = nn.Sequential(
            nn.Linear(2048, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, proj_dim, bias=False),
        )

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor],
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Runs a single training step for contrastive learning.

        Args:
            batch: A tuple of two augmented views of input tensors.
            optimizer: The optimizer to use.
            loss_fn: Contrastive loss function (e.g., NT-Xent).
            device: Torch device to run the computation on.

        Returns:
            Dictionary with training loss.
        """
        x_i, x_j = batch
        x_i, x_j = x_i.to(device), x_j.to(device)
        optimizer.zero_grad()
        z_i, z_j = self(x_i), self(x_j)
        loss = loss_fn(z_i, z_j)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor],
        loss_fn: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Runs a single validation step for contrastive loss.

        Args:
            batch: A tuple of two augmented views of input tensors.
            loss_fn: Contrastive loss function.
            device: Torch device.

        Returns:
            Dictionary with validation loss.
        """
        x_i, x_j = batch
        x_i, x_j = x_i.to(device), x_j.to(device)
        z_i, z_j = self(x_i), self(x_j)
        loss = loss_fn(z_i, z_j)
        return {"val_loss": loss.item()}

    def encode(self, x: Tensor) -> Tensor:
        """Encodes input using the backbone network."""
        return self.encoder(x)  # type: ignore[no-any-return]

    def project(self, h: Tensor) -> Tensor:
        """Applies the projection head to the encoded features."""
        return self.projector(h)  # type: ignore[no-any-return]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder and projector."""
        h = self.encode(x)
        z = self.project(h)
        return F.normalize(z, dim=1)
