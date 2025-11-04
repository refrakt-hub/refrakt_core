# wrappers/autoencoder.py

from typing import Any

import torch
from torch import nn

from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("autoencoder")
class AutoencoderWrapper(nn.Module):
    def __init__(self, model: nn.Module, variant: str = "simple") -> None:
        super().__init__()
        self.backbone = model
        self.variant = variant

    def forward(self, x: torch.Tensor) -> ModelOutput:
        output: Any = self.backbone(x)

        if self.variant == "vae":
            return ModelOutput(
                reconstruction=output["recon"],
                extra={
                    "mu": output["mu"],
                    "logvar": output["logvar"],
                },
            )
        elif self.variant == "mae":
            return ModelOutput(
                reconstruction=output["recon"],
                extra={
                    "mask": output["mask"],
                    "original_patches": output["original_patches"],
                },
            )
        else:
            # Handle both dictionary and tensor outputs for simple variant
            if isinstance(output, dict) and "recon" in output:
                return ModelOutput(reconstruction=output["recon"])
            else:
                return ModelOutput(reconstruction=output)

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.forward(x).reconstruction
        if reconstruction is None:
            raise ValueError("Reconstruction is None")
        return (
            torch.as_tensor(reconstruction)
            if not isinstance(reconstruction, torch.Tensor)
            else reconstruction
        )

    def get_latents_and_labels(self, batch):
        # Support dict, tuple/list, or tensor batch
        if isinstance(batch, dict):
            x = batch.get("input", None)
            labels = batch.get("label", None)
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        else:
            x = batch
            labels = None

        if x is None:
            raise ValueError(
                "Input tensor 'x' not found in batch for get_latents_and_labels."
            )
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        # Get latent from backbone
        with torch.no_grad():
            latents = self.backbone.get_latent(x.to(next(self.parameters()).device))
        latents_np = latents.detach().cpu().numpy()
        labels_np = (
            labels.detach().cpu().numpy()
            if labels is not None and hasattr(labels, "detach")
            else labels
        )
        return latents_np, labels_np

    def get_disentanglement_latent(self, batch):
        # Extract a single input (first sample) from the batch
        if isinstance(batch, dict):
            x = batch.get("input", None)
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        if x is None:
            raise ValueError(
                "Input tensor 'x' not found in batch for get_disentanglement_latent."
            )
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        # Use only the first sample in the batch
        x_single = x[0].unsqueeze(0) if x.ndim > 1 else x.unsqueeze(0)

        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            latent = self.backbone.get_latent(
                x_single.to(next(self.parameters()).device)
            )
        if was_training:
            self.backbone.train()
        # If latent is a tuple (e.g., (mu, sigma)), use mu
        if isinstance(latent, tuple):
            latent = latent[0]
        return latent.squeeze().detach().cpu().numpy()

    def decode(self, z):
        return self.backbone.decode(z)

    def get_saliency_input(self, batch):
        # Extract the input image from the batch for saliency visualization
        if isinstance(batch, dict):
            x = batch.get("input", None)
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        if x is None:
            raise ValueError(
                "Input tensor 'x' not found in batch for get_saliency_input."
            )
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        return x

    def get_inputs_and_recons(self, batch):
        # Extract input tensor from batch
        if isinstance(batch, dict):
            x = batch.get("input", None)
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        if x is None:
            raise ValueError(
                "Input tensor 'x' not found in batch for get_inputs_and_recons."
            )
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.to(next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            recon = output.reconstruction
            if recon is None:
                raise ValueError(
                    "Model output does not contain a reconstruction for get_inputs_and_recons."
                )
        return x.detach().cpu().numpy(), recon.detach().cpu().numpy()

    def generate_samples(self, batch):
        # Generate random samples from the latent space and decode
        import torch

        batch_size = None
        if isinstance(batch, dict):
            x = batch.get("input", None)
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        if x is not None and hasattr(x, "shape"):
            batch_size = x.shape[0]
        else:
            batch_size = 8  # fallback
        # Get latent dim
        hidden_dim = getattr(self, "hidden_dim", None)
        if hidden_dim is None and hasattr(self.backbone, "hidden_dim"):
            hidden_dim = self.backbone.hidden_dim
        if isinstance(hidden_dim, torch.Tensor):
            hidden_dim = int(hidden_dim.item())
        elif not isinstance(hidden_dim, int):
            try:
                hidden_dim = int(hidden_dim)
            except Exception:
                hidden_dim = 32  # fallback
        if hidden_dim is None or isinstance(hidden_dim, torch.nn.Module):
            hidden_dim = 32  # fallback
        if not isinstance(hidden_dim, int):
            hidden_dim = 32
        device = next(self.parameters()).device
        z = torch.randn(batch_size, hidden_dim, device=device)
        self.eval()
        with torch.no_grad():
            samples = self.decode(z)
        return samples.detach().cpu().numpy()

    def get_latent(self, x):
        return self.backbone.get_latent(x)
