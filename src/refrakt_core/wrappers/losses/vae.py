# wrappers/vae.py

from typing import Any, Dict, Optional, Union

from torch import Tensor, nn

from refrakt_core.losses.vae import VAELoss
from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.utils.vae_loss_utils import (
    compute_reconstruction_loss,
    create_vae_loss_output,
    extract_vae_components,
)


@register_loss("vae_wrapped", mode="reconstruction")
class VAELossWrapper(nn.Module):
    def __init__(self, loss_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        loss_params = loss_params or {}
        self.loss_fn = VAELoss(**loss_params)
        self.kld_weight = self.loss_fn.kld_weight
        self.recon_loss_type = self.loss_fn.recon_loss_type

    def forward(
        self, output: Union[ModelOutput, Dict[str, Any], Tensor], target: Tensor
    ) -> LossOutput:
        # Extract components
        recon, mu, logvar = extract_vae_components(output)

        # Reshape target to match reconstruction shape
        target = target.view(recon.shape)

        recon_flat = recon.view(recon.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Compute full loss from base VAELoss
        total_loss = self.loss_fn(
            (
                {"recon": recon_flat, "mu": mu, "logvar": logvar}
                if mu is not None and logvar is not None
                else recon_flat
            ),
            target_flat,
        )

        # Compute reconstruction loss for logging
        recon_loss = compute_reconstruction_loss(
            recon_flat, target_flat, self.recon_loss_type
        )

        return create_vae_loss_output(total_loss, recon_loss, mu, logvar)
