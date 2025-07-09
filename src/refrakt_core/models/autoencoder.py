"""
AutoEncoder implementation with optional VAE support.

Defines a basic feed-forward auto-encoder or a variational auto-encoder (VAE),
depending on the selected mode / type.
"""

from typing import Any, Tuple, Union, cast

import torch
from torch import Tensor, nn

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
        mode: str | None = None,  # preferred name
        variant: str | None = None,  # noqa: A002  (keep for backward-compat)
        model_name: str = "autoencoder",
    ) -> None:
        # allow either keyword; prefer `mode`
        chosen = (mode or variant or "simple").lower()
        if chosen not in {"simple", "vae"}:
            raise ValueError(f"Unsupported autoencoder mode/type: {chosen!r}")

        super().__init__(hidden_dim=hidden_dim, model_name=model_name)

        # expose both attributes so that legacy code & new code work
        self.mode: str = chosen
        self.variant: str = chosen

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.input_shape = (1, 28, 28) if input_dim == 784 else (input_dim,)

        # ── encoder / decoder ───────────────────────────────────────────────
        if self.input_dim == 784:
            self.encoder_layers = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Linear(128, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.decoder_layers = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Linear(512, input_dim),
                nn.Sigmoid(),
            )

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

        else:
            in_channels = 1 if input_dim == 224 * 224 else 3

            self.encoder_layers = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 56 * 56, hidden_dim),
            )

            self.decoder_layers = nn.Sequential(
                nn.Linear(hidden_dim, 32 * 56 * 56),
                nn.Unflatten(1, (32, 56, 56)),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    16, in_channels, 3, stride=2, padding=1, output_padding=1
                ),
                nn.Sigmoid(),
            )

        if self.mode == "vae":
            self.mu = nn.Linear(hidden_dim, hidden_dim)
            self.sigma = nn.Linear(hidden_dim, hidden_dim)

    # ──────────────────────────────────────────────────────────────────────
    # forward helpers
    # ──────────────────────────────────────────────────────────────────────
    def encode(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        encoded = cast(Tensor, self.encoder_layers(x))
        if self.mode == "vae":
            mu, sigma = self.mu(encoded), self.sigma(encoded)
            return mu, sigma
        return encoded

    def decode(self, z: Any) -> Tensor:
        return self.decoder_layers(z)  # type: ignore[no-any-return]

    @staticmethod
    def _reparameterize(mu: Tensor, sigma: Tensor) -> Tensor:
        std = torch.exp(0.5 * sigma)
        return mu + torch.randn_like(std) * std

    def get_latent(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Return latent representation (μ for VAE, encoded vector otherwise)"""
        if self.mode == "vae":
            mu, _ = self.encode(x)
            return mu
        else:
            return self.encode(x)

    # def training_step(
    #     self, batch: Tuple[Tensor, ...], optimizer: Optimizer,
    #     loss_fn: nn.Module, device: torch.device,
    # ) -> Dict[str, float]:
    #     inputs = batch[0].to(device)
    #     optimizer.zero_grad()
    #     output = self(inputs)

    #     if self.mode == "vae":                          # VAE loss = MSE + KL
    #         recon, mu, logvar = output["recon"], output["mu"], output["logvar"]
    #         mse = loss_fn(recon, inputs)
    #         kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #         loss = mse + kl
    #     else:                                           # plain MSE
    #         loss = loss_fn(output, inputs)

    #     loss.backward()
    #     optimizer.step()
    #     return {"loss": loss.item()}

    # def validation_step(
    #     self, batch: Tuple[Tensor, ...], loss_fn: nn.Module,
    #     device: torch.device,
    # ) -> Dict[str, float]:
    #     inputs = batch[0].to(device)
    #     output = self(inputs)

    #     if self.mode == "vae":
    #         recon, mu, logvar = output["recon"], output["mu"], output["logvar"]
    #         mse = loss_fn(recon, inputs)
    #         kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #         loss = mse + kl
    #     else:
    #         loss = loss_fn(output, inputs)
    #     return {"val_loss": loss.item()}

    # ──────────────────────────────────────────────────────────────────────
    # autoencoder.py - Update the forward method
    def forward(self, x: Tensor) -> Any:
        # Store original shape
        original_shape = x.shape

        # Flatten if needed (for linear models)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        if self.mode == "simple":
            encoded = self.encode(x)
            decoded = self.decode(encoded)
            return decoded.view(original_shape)  # Reshape to original input dimensions

        # VAE forward
        mu, sigma = self.encode(x)
        z = self._reparameterize(mu, sigma)
        decoded = self.decode(z)
        recon = decoded.view(original_shape)  # Reshape to original input dimensions
        logvar = torch.log(sigma.pow(2) + 1e-7)
        return {"recon": recon, "mu": mu, "logvar": logvar}
