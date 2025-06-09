"""
Loss implementation for the masked autoencoder.
"""

from typing import Dict

import torch
from torch import Tensor

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss


@register_loss("mae")
class MAELoss(BaseLoss):
    """
    Masked Autoencoder Loss.

    Computes MSE only over masked patches, following the methodology in the MAE paper.
    Optionally normalizes target patches before computing the loss.

    Args:
        normalize_target (bool): Whether to normalize the original patches before loss computation.
    """

    def __init__(self, normalize_target: bool = False) -> None:
        super().__init__(name="MAELoss")
        self.normalize_target: bool = normalize_target

    def forward(self, predictions: Dict[str, Tensor], targets: Tensor = None) -> Tensor:
        """
        Compute reconstruction loss over masked patches.

        Args:
            predictions (dict): Dictionary with keys:
                - "recon_patches" (Tensor): Reconstructed patches of shape (B, N, patch_dim).
                - "mask" (Tensor): Binary mask tensor of shape (B, N) indicating masked patches.
                - "original_patches" (Tensor): Ground truth patches of shape (B, N, patch_dim).
            targets (Tensor, optional): Not used for MAE loss since targets are in predictions dict.

        Returns:
            Tensor: Scalar loss value representing masked MSE.
        """
        
        pred: Tensor = predictions["recon"]
        mask: Tensor = predictions["mask"].unsqueeze(-1)  # (B, N, 1)
        original: Tensor = predictions["original_patches"]

        if self.normalize_target:
            mean = original.mean(dim=-1, keepdim=True)
            std = original.std(dim=-1, keepdim=True) + 1e-6
            original = (original - mean) / std

        loss: Tensor = ((pred - original) ** 2) * mask
        return loss.sum() / mask.sum()

    def get_config(self) -> Dict[str, bool]:
        """
        Return the configuration of the MAELoss.

        Returns:
            dict: Dictionary containing loss configuration.
        """
        config = super().get_config()
        config.update({"normalize_target": self.normalize_target})
        return config

    def extra_repr(self) -> str:
        """
        String representation for printing.

        Returns:
            str: Informative string for debugging/logging.
        """
        return f"name={self.name}, normalize_target={self.normalize_target}"