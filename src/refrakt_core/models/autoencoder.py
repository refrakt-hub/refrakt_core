"""
AutoEncoder implementation with optional VAE support.

Defines a basic feed-forward auto-encoder or a variational auto-encoder (VAE),
depending on the selected mode / type.
"""

from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from refrakt_core.models.templates.models import BaseAutoEncoder  # unchanged
from refrakt_core.registry.model_registry import register_model


@register_model("autoencoder")
class AutoEncoder(BaseAutoEncoder):
    """
    Feed-forward AutoEncoder and Variational AutoEncoder (VAE).

    Args:
        input_dim: Flattened input dimension (e.g. 784 for 28×28 images).
        hidden_dim: Latent dimension.
        mode / type: 'simple' or 'vae'.
        model_name: Name used for registry / logging.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 8,
        mode: str | None = None,           # preferred name
        type: str | None = None,           # noqa: A002  (keep for backward-compat)
        model_name: str = "autoencoder",
    ) -> None:
        # allow either keyword; prefer `mode`
        chosen = (mode or type or "simple").lower()
        if chosen not in {"simple", "vae"}:
            raise ValueError(f"Unsupported autoencoder mode/type: {chosen!r}")

        super().__init__(hidden_dim=hidden_dim, model_name=model_name)

        # expose both attributes so that legacy code & new code work
        self.mode: str = chosen
        self.type: str = chosen

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim

        # ── encoder / decoder ───────────────────────────────────────────────
        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64),        nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.ReLU(inplace=True),
        )
        self.decoder_layers = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 256),        nn.ReLU(inplace=True),
            nn.Linear(256, input_dim), nn.Sigmoid(),
        )

        if self.mode == "vae":
            self.mu    = nn.Linear(hidden_dim, hidden_dim)
            self.sigma = nn.Linear(hidden_dim, hidden_dim)

    # ──────────────────────────────────────────────────────────────────────
    # forward helpers
    # ──────────────────────────────────────────────────────────────────────
    def encode(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        encoded = self.encoder_layers(x)
        if self.mode == "vae":
            mu, sigma = self.mu(encoded), self.sigma(encoded)
            return mu, sigma
        return encoded

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder_layers(z)

    @staticmethod
    def _reparameterize(mu: Tensor, sigma: Tensor) -> Tensor:
        std = torch.exp(0.5 * sigma)
        return mu + torch.randn_like(std) * std

    def get_latent(self, x: Tensor) -> Tensor:
        """
        Return the latent representation (μ for VAE, encoded vector otherwise).
        """
        enc = self.encode(x)
        return enc[0] if self.mode == "vae" else enc

  
    def training_step(
        self, batch: Tuple[Tensor, ...], optimizer: Optimizer,
        loss_fn: nn.Module, device: torch.device,
    ) -> Dict[str, float]:
        inputs = batch[0].to(device)
        optimizer.zero_grad()
        output = self(inputs)

        if self.mode == "vae":                          # VAE loss = MSE + KL
            recon, mu, logvar = output["recon"], output["mu"], output["logvar"]
            mse = loss_fn(recon, inputs)
            kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + kl
        else:                                           # plain MSE
            loss = loss_fn(output, inputs)

        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def validation_step(
        self, batch: Tuple[Tensor, ...], loss_fn: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        inputs = batch[0].to(device)
        output = self(inputs)

        if self.mode == "vae":
            recon, mu, logvar = output["recon"], output["mu"], output["logvar"]
            mse = loss_fn(recon, inputs)
            kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + kl
        else:
            loss = loss_fn(output, inputs)
        return {"val_loss": loss.item()}

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        if self.mode == "simple":
            return self.decode(self.encode(x))          # type: ignore[arg-type]

        # VAE forward
        mu, sigma = self.encode(x)                      # type: ignore[misc]
        z      = self._reparameterize(mu, sigma)
        recon  = self.decode(z)
        logvar = torch.log(sigma.pow(2) + 1e-7)
        return {"recon": recon, "mu": mu, "logvar": logvar}
