"""
Utility functions for VAE loss wrappers.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


def extract_vae_components(
    output: Union[ModelOutput, Dict[str, Any], Tensor],
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Extract reconstruction, mu, and logvar from output."""
    if isinstance(output, ModelOutput):
        recon = output.reconstruction
        mu = output.extra.get("mu") if hasattr(output, "extra") else None
        logvar = output.extra.get("logvar") if hasattr(output, "extra") else None
    elif isinstance(output, dict):
        recon = output.get("recon")
        mu = output.get("mu")
        logvar = output.get("logvar")

        # Fallback to 'reconstruction' key in dict
        if recon is None:
            recon = output.get("reconstruction")
    else:
        recon = output
        mu = logvar = None

    if recon is None:
        # Last resort: try to access output directly
        if isinstance(output, torch.Tensor):
            recon = output
        else:
            raise ValueError(
                "[VAELossWrapper] Could not find reconstruction tensor in model output"
            )

    return recon, mu, logvar


def compute_reconstruction_loss(
    recon_flat: Tensor, target_flat: Tensor, recon_loss_type: str
) -> Tensor:
    """Compute reconstruction loss based on type."""
    if recon_loss_type == "mse":
        return nn.functional.mse_loss(recon_flat, target_flat, reduction="sum")
    elif recon_loss_type == "l1":
        return nn.functional.l1_loss(recon_flat, target_flat, reduction="sum")
    else:
        raise ValueError(f"[VAELossWrapper] Invalid recon_loss_type: {recon_loss_type}")


def compute_kld_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    """Compute KL divergence loss."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def create_vae_loss_output(
    total_loss: Tensor,
    recon_loss: Tensor,
    mu: Optional[Tensor],
    logvar: Optional[Tensor],
) -> LossOutput:
    """Create LossOutput with appropriate components."""
    if mu is None or logvar is None:
        return LossOutput(total=total_loss, components={"recon_loss": recon_loss})

    kld_loss = compute_kld_loss(mu, logvar)
    return LossOutput(
        total=total_loss,
        components={
            "recon_loss": recon_loss.detach(),
            "kld_loss": kld_loss.detach(),
        },
    )
