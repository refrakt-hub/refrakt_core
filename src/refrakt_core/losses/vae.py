"""
VAE Loss Module which includes reconstruction (MSE or L1) and KL Divergence Loss.
"""

from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from refrakt_core.registry.loss_registry import register_loss


@register_loss("vae")
class VAELoss(nn.Module):
    """
    Variational Autoencoder (VAE) loss combining:
    - Reconstruction loss (MSE or L1).
    - KL divergence between latent distribution and standard normal prior.

    Can also function as a standard autoencoder loss when latent stats (mu/logvar) are absent.

    Args:
        recon_loss_type (str): Type of reconstruction loss to use. Options: 'mse', 'l1'.
        kld_weight (float): Scaling factor for the KL divergence term.
    """

    def __init__(self, recon_loss_type: str = "mse", kld_weight: float = 1.0) -> None:
        super().__init__()
        self.recon_loss_type = recon_loss_type
        self.kld_weight = kld_weight
        self.normalize = True

    def forward(
        self, model_output: Union[Tensor, Dict[str, Tensor]], target: Tensor
    ) -> Tensor:
        """
        Compute the VAE loss.

        Args:
            model_output (Tensor or Dict): Either the reconstructed tensor (Tensor)
                                           or a dict with keys:
                                             - "recon": Reconstructed tensor
                                             - "mu": Mean of latent distribution
                                             - "logvar": Log variance of latent distribution
            target (Tensor): Ground truth tensor.

        Returns:
            Tensor: Total loss (reconstruction + KL divergence if applicable).

        Raises:
            ValueError: If `recon_loss_type` is not one of {'mse', 'l1'}.
        """
        if isinstance(model_output, dict):
            recon: Tensor = model_output["recon"]
            mu: Optional[Tensor] = model_output.get("mu")
            logvar: Optional[Tensor] = model_output.get("logvar")
        else:
            recon = model_output
            mu, logvar = None, None

        # Reconstruction loss
        recon_flat = recon.view(recon.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        num_elements = torch.prod(torch.tensor(recon.shape[1:]) * recon.shape[0])

        # Reconstruction loss using flattened tensors
        if self.recon_loss_type == "mse":
            recon_loss: Tensor = F.mse_loss(recon_flat, target_flat, reduction="sum")
            if self.normalize:
                recon_loss = recon_loss / num_elements
        elif self.recon_loss_type == "l1":
            recon_loss = F.l1_loss(recon_flat, target_flat, reduction="sum")
            if self.normalize:
                recon_loss = recon_loss / num_elements
        else:
            raise ValueError(
                f"Unsupported reconstruction loss type: '{self.recon_loss_type}'"
            )

        # Only return recon loss if not a VAE
        if mu is None or logvar is None:
            return recon_loss

        # KL Divergence loss
        kld_loss: Tensor = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + self.kld_weight * kld_loss
