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
    def __init__(self, temp_anchor=0.1, temp_target=0.04, lambda_me_max=1.0) -> None:
        super().__init__()
        self.temp_anchor = temp_anchor
        self.temp_target = temp_target
        self.lambda_me_max = lambda_me_max

    def forward(self, z_anchor: Tensor, z_target: Tensor, prototypes: Tensor) -> Tensor:
        if z_anchor.ndim != 2 or z_target.ndim != 2 or prototypes.ndim != 2:
            raise ValueError("All inputs must be 2D tensors.")

        if z_anchor.shape[1] != z_target.shape[1] or z_anchor.shape[1] != prototypes.shape[1]:
            raise ValueError("Feature dimensions (D) must match for all inputs.")

        if z_anchor.shape[0] % z_target.shape[0] != 0:
            raise ValueError("Anchor batch size must be a multiple of target batch size.")

        B = z_target.shape[0]
        M = z_anchor.shape[0] // B

        # Compute logits
        logits_anchor = torch.matmul(z_anchor, prototypes.T) / self.temp_anchor  # (B*M, K)
        logits_target = torch.matmul(z_target, prototypes.T) / self.temp_target  # (B, K)

        # Compute soft targets
        p_target = F.softmax(logits_target, dim=-1).repeat_interleave(M, dim=0)  # (B*M, K)
        logp_anchor = F.log_softmax(logits_anchor, dim=-1)                       # (B*M, K)

        # KL divergence between anchor prediction and target distribution
        loss_ce = F.kl_div(logp_anchor, p_target.detach(), reduction="batchmean")

        # Entropy of mean anchor distribution
        p_anchor = F.softmax(logits_anchor, dim=-1)
        p_mean = p_anchor.mean(dim=0)
        entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-6))
        loss_entropy = -self.lambda_me_max * entropy

        return loss_ce + loss_entropy

    def get_config(self) -> Dict[str, float]:
        return {
            "temp_anchor": self.temp_anchor,
            "temp_target": self.temp_target,
            "lambda_me_max": self.lambda_me_max,
        }
