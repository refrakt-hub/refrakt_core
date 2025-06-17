"""
MSNLossWrapper: A wrapper class for the Masked Siamese Network (MSN) loss.
"""

import torch
from torch import nn
from typing import Dict, Optional

from refrakt_core.losses.msn import MSNLoss
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.registry.loss_registry import register_loss


@register_loss("msn", mode="embedding")
class MSNLossWrapper(nn.Module):
    """
    Class-based wrapper around the MSNLoss that returns standardized LossOutput.

    Args:
        loss_params (dict): Dictionary containing temp_anchor, temp_target, lambda_me_max.
    """

    def __init__(self, loss_params: Optional[Dict] = None):
        super().__init__()
        loss_params = loss_params or {}

        self.loss_fn = MSNLoss(
            temp_anchor=loss_params.get("temp_anchor", 0.1),
            temp_target=loss_params.get("temp_target", 0.04),
            lambda_me_max=loss_params.get("lambda_me_max", 1.0)
        )

    def forward(self, output: ModelOutput, target=None) -> LossOutput:
        """
        Args:
            output (ModelOutput): The structured model output containing embeddings, extra fields.
            target (Unused): Included for compatibility.

        Returns:
            LossOutput: Contains total loss and component breakdown.
        """
        z_anchor = output.embeddings  # shape: (B*M, D)
        z_target = output.extra.get("target_embeddings")  # shape: (B, D)
        prototypes = output.extra.get("prototypes")       # shape: (K, D)

        if z_anchor is None or z_target is None or prototypes is None:
            raise ValueError("Missing required fields: embeddings, target_embeddings, or prototypes.")

        loss_ce, loss_entropy = self._compute_components(z_anchor, z_target, prototypes)
        total_loss = loss_ce + loss_entropy

        return LossOutput(
            total=total_loss,
            components={
                "kl_div": loss_ce,
                "entropy_reg": loss_entropy
            }
        )

    def _compute_components(self, z_anchor, z_target, prototypes):
        B = z_target.shape[0]
        M = z_anchor.shape[0] // B

        # Compute logits
        logits_anchor = torch.matmul(z_anchor, prototypes.T) / self.loss_fn.temp_anchor
        logits_target = torch.matmul(z_target, prototypes.T) / self.loss_fn.temp_target

        # Soft targets
        p_target = torch.nn.functional.softmax(logits_target, dim=-1).repeat_interleave(M, dim=0)
        logp_anchor = torch.nn.functional.log_softmax(logits_anchor, dim=-1)

        # KL divergence
        loss_ce = torch.nn.functional.kl_div(logp_anchor, p_target.detach(), reduction="batchmean")

        # Entropy of mean prediction
        p_anchor = torch.nn.functional.softmax(logits_anchor, dim=-1)
        p_mean = p_anchor.mean(dim=0)
        entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-6))
        loss_entropy = -self.loss_fn.lambda_me_max * entropy

        return loss_ce, loss_entropy
