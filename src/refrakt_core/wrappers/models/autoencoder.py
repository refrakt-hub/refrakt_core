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
