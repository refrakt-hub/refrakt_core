"""
Masked Siamese Network (MSN) Loss Implementation.
"""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss


@register_loss("msn")
class MSNLoss(BaseLoss):
    """
    Loss function for Masked Siamese Networks (MSN).

    Combines:
    - Cross-entropy loss between predicted and target prototype distributions.
    - Entropy maximization to encourage uniform usage of all prototypes.

    Reference:
        Assran et al., "Masked Siamese Networks for Label-Efficient Learning", CVPR 2023.

    Args:
        temp_anchor (float): Temperature for scaling anchor-prototype logits.
        temp_target (float): Temperature for scaling target-prototype logits.
        lambda_me_max (float): Coefficient for entropy maximization.
    """

    def __init__(self, temp_anchor: float = 0.1, temp_target: float = 0.04, lambda_me_max: float = 1.0) -> None:
        super().__init__()
        self.temp_anchor: float = temp_anchor
        self.temp_target: float = temp_target
        self.lambda_me_max: float = lambda_me_max

    def forward(
        self,
        z_anchor: Tensor,
        z_target: Tensor,
        prototypes: Tensor
    ) -> Tensor:
        """
        Compute MSN loss.

        Args:
            z_anchor (Tensor): Anchor embeddings (B * M, D), from masked views.
            z_target (Tensor): Target embeddings (B, D), from unmasked views.
            prototypes (Tensor): Prototype vectors (K, D).

        Returns:
            Tensor: Scalar tensor representing the combined MSN loss.

        Raises:
            ValueError: If shapes or dimensions are inconsistent.
        """
        if z_anchor.ndim != 2 or z_target.ndim != 2 or prototypes.ndim != 2:
            raise ValueError("All inputs must be 2D tensors.")

        if z_anchor.shape[1] != z_target.shape[1] or z_anchor.shape[1] != prototypes.shape[1]:
            raise ValueError("Feature dimensions (D) must match for all inputs.")

        if z_anchor.shape[0] % z_target.shape[0] != 0:
            raise ValueError("Anchor batch size must be a multiple of target batch size.")

        B: int = z_target.shape[0]
        M: int = z_anchor.shape[0] // B

        logits_anchor: Tensor = torch.matmul(z_anchor, prototypes.T) / self.temp_anchor  # (B*M, K)
        logits_target: Tensor = torch.matmul(z_target, prototypes.T) / self.temp_target  # (B, K)

        # Convert to probability distributions
        p_anchor: Tensor = F.softmax(logits_anchor, dim=-1)  # (B*M, K)
        p_target: Tensor = F.softmax(logits_target, dim=-1).repeat_interleave(M, dim=0)  # (B*M, K)

        # Cross-entropy loss between predicted and target soft distributions
        loss_ce: Tensor = F.cross_entropy(
            p_anchor.log(), p_target.detach(), reduction="none"
        ).mean()

        # Entropy maximization on mean anchor distribution
        p_mean: Tensor = p_anchor.mean(dim=0)
        entropy: Tensor = -torch.sum(p_mean * torch.log(p_mean + 1e-6))
        loss_entropy: Tensor = -self.lambda_me_max * entropy

        return loss_ce + loss_entropy

    def get_config(self) -> Dict[str, float]:
        """
        Return the configuration of the MSN loss.

        Returns:
            dict: Loss configuration.
        """
        return {
            "temp_anchor": self.temp_anchor,
            "temp_target": self.temp_target,
            "lambda_me_max": self.lambda_me_max,
        }
