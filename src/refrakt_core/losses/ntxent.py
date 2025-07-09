"""
NT-Xent Loss Implementation for contrastive learning.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss


@register_loss("nt_xent")
class NTXentLoss(BaseLoss):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.

    Used in contrastive learning frameworks like SimCLR. Encourages positive
    pairs to be closer while separating all other (negative) pairs.

    Args:
        temperature (float): Temperature scaling factor.
        name (str, optional): Name of the loss (defaults to class name).
    """

    def __init__(self, temperature: float = 0.5, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.temperature: float = temperature

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Compute the NT-Xent loss.

        Args:
            z1 (Tensor): First batch of embeddings with shape (N, D).
            z2 (Tensor): Second batch of embeddings with shape (N, D).

        Returns:
            Tensor: Scalar loss value.

        Raises:
            ValueError: If batch size is less than or equal to 1.
        """
        n: int = z1.size(0)
        if n <= 1:
            raise ValueError("Batch size must be > 1 for NT-Xent loss.")

        z: Tensor = torch.cat([z1, z2], dim=0)  # (2N, D)
        z_norm: Tensor = F.normalize(z, dim=1)  # L2 normalization

        sim_matrix: Tensor = (
            torch.matmul(z_norm, z_norm.T) / self.temperature
        )  # (2N, 2N)

        # Mask self-similarity
        mask: Tensor = torch.eye(2 * n, device=z.device).bool()
        sim_matrix.masked_fill_(mask, float("-inf"))

        # Positive pairs: i-th and (i+n)-th (for i in [0, N))
        positive_indices: Tensor = torch.cat(
            [torch.arange(n, 2 * n), torch.arange(0, n)]
        ).to(z.device)
        pos_sim: Tensor = sim_matrix[torch.arange(2 * n), positive_indices]

        # Compute contrastive loss
        exp_sim: Tensor = torch.exp(sim_matrix)
        denom: Tensor = exp_sim.sum(dim=1)

        loss: Tensor = -torch.log(torch.exp(pos_sim) / denom)
        return loss.mean()

    def get_config(self) -> Dict[str, object]:
        """
        Return the configuration of the loss function.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update({"temperature": self.temperature, "type": "contrastive"})
        return config

    def extra_repr(self) -> str:
        """
        Return a string representation for printing/debugging.

        Returns:
            str: Loss configuration summary.
        """
        return f"name={self.name}, temperature={self.temperature}"
