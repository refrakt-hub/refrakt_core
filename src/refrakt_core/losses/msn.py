"""
Masked Siamese Network (MSN) Loss Implementation.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss


@register_loss("msn")
class MSNLoss(BaseLoss):
    def __init__(
        self,
        temp_anchor: float = 0.1,
        temp_target: float = 0.04,
        lambda_me_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.temp_anchor = temp_anchor
        self.temp_target = temp_target
        self.lambda_me_max = lambda_me_max
        print(
            f"[INIT] MSNLossWrapper initialized with temp_anchor={temp_anchor}, temp_target={temp_target}, lambda_me_max={lambda_me_max}"
        )

    def forward(self, z_anchor: Tensor, z_target: Tensor, prototypes: Tensor) -> Tensor:
        total_loss, _ = self.compute_with_components(z_anchor, z_target, prototypes)
        return total_loss

    def compute_with_components(
        self, z_anchor: Tensor, z_target: Tensor, prototypes: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if z_anchor.ndim != 2 or z_target.ndim != 2 or prototypes.ndim != 2:
            raise ValueError("All inputs must be 2D tensors.")

        if (
            z_anchor.shape[1] != z_target.shape[1]
            or z_anchor.shape[1] != prototypes.shape[1]
        ):
            raise ValueError("Feature dimensions (D) must match for all inputs.")

        if z_anchor.shape[0] % z_target.shape[0] != 0:
            raise ValueError(
                "Anchor batch size must be a multiple of target batch size."
            )

        B = z_target.shape[0]
        M = z_anchor.shape[0] // B

        logits_anchor = torch.matmul(z_anchor, prototypes.T) / self.temp_anchor
        logits_target = torch.matmul(z_target, prototypes.T) / self.temp_target

        p_target = F.softmax(logits_target, dim=-1).repeat_interleave(M, dim=0)
        logp_anchor = F.log_softmax(logits_anchor, dim=-1)

        loss_ce = F.kl_div(logp_anchor, p_target.detach(), reduction="batchmean")

        p_anchor = F.softmax(logits_anchor, dim=-1)
        p_mean = p_anchor.mean(dim=0)
        entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-6))
        loss_entropy = -self.lambda_me_max * entropy

        total_loss = loss_ce + loss_entropy

        components = {
            "loss_ce": loss_ce.detach(),
            "loss_entropy": loss_entropy.detach(),
            "total": total_loss.detach(),
        }

        return total_loss, components
