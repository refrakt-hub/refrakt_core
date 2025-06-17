# wrappers/vae.py

from typing import Dict, Optional, Union
import torch
from torch import nn, Tensor
from refrakt_core.losses.vae import VAELoss
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.loss_registry import register_loss


@register_loss("vae_wrapped", mode="reconstruction")
class VAELossWrapper(nn.Module):
    def __init__(self, loss_params: Optional[Dict] = None):
        super().__init__()
        loss_params = loss_params or {}
        self.loss_fn = VAELoss(**loss_params)
        self.kld_weight = self.loss_fn.kld_weight
        self.recon_loss_type = self.loss_fn.recon_loss_type

    def forward(self, output: Union[ModelOutput, Dict, Tensor], target: Tensor) -> LossOutput:
        # First, try to access reconstruction directly if it's a ModelOutput
        if hasattr(output, 'reconstruction'):
            recon = output.reconstruction
            mu = output.extra.get("mu") if hasattr(output, 'extra') else None
            logvar = output.extra.get("logvar") if hasattr(output, 'extra') else None
        elif isinstance(output, dict):
            recon = output.get("recon")
            mu = output.get("mu")
            logvar = output.get("logvar")
        else:
            recon = output
            mu = logvar = None

        # FIX: Add fallback to 'reconstruction' key in dict
        if recon is None and isinstance(output, dict):
            recon = output.get("reconstruction")
            
        if recon is None:
            # Last resort: try to access output directly
            if isinstance(output, torch.Tensor):
                recon = output
            else:
                raise ValueError("[VAELossWrapper] Could not find reconstruction tensor in model output")

        # FIX: Reshape target to match reconstruction shape
        target = target.view(recon.shape)

        recon_flat = recon.view(recon.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        # Compute full loss from base VAELoss
        total_loss = self.loss_fn(
            {"recon": recon_flat, "mu": mu, "logvar": logvar} if mu is not None and logvar is not None else recon_flat,
            target_flat
        )

        # Split components for logging
        if self.recon_loss_type == "mse":
            recon_loss = nn.functional.mse_loss(recon_flat, target_flat, reduction="sum")
        elif self.recon_loss_type == "l1":
            recon_loss = nn.functional.l1_loss(recon_flat, target_flat, reduction="sum")
        else:
            raise ValueError(f"[VAELossWrapper] Invalid recon_loss_type: {self.recon_loss_type}")

        if mu is None or logvar is None:
            return LossOutput(total=total_loss, components={"recon_loss": recon_loss})

        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return LossOutput(
            total=total_loss,
            components={
                "recon_loss": recon_loss.detach(),
                "kld_loss": kld_loss.detach()
            }
        )