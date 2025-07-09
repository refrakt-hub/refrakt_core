"""
Loss implementation for the masked autoencoder.
"""

from typing import Dict, Optional

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

    def __init__(self, normalize_target: bool = False, patch_size: int = 16) -> None:
        super().__init__(name="MAELoss")
        self.normalize_target: bool = normalize_target
        self.patch_size = patch_size

    def patchify(self, imgs: Tensor) -> Tensor:
        """
        Split reconstructed images into patch sequences.
        Args:
            imgs (Tensor): (B, C, H, W)
        Returns:
            Tensor: (B, N, patch_dim)
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0

        h = H // p
        w = W // p
        return (
            imgs.reshape(B, C, h, p, w, p)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(B, h * w, p * p * C)
        )

    def forward(
        self, predictions: Dict[str, Tensor], targets: Optional[Tensor] = None
    ) -> Tensor:
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
        pred: Tensor = predictions["recon"]  # (B, C, H, W)
        mask: Tensor = predictions["mask"].unsqueeze(-1)  # (B, N, 1)
        original: Tensor = predictions["original_patches"]  # (B, N, patch_dim)

        pred_patches = self.patchify(pred)  # convert to (B, N, patch_dim)

        if self.normalize_target:
            mean = original.mean(dim=-1, keepdim=True)
            std = original.std(dim=-1, keepdim=True) + 1e-6
            original = (original - mean) / std

        loss: Tensor = ((pred_patches - original) ** 2) * mask
        return loss.sum() / mask.sum()

    def get_config(self) -> dict[str, bool]:
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
