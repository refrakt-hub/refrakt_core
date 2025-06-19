# wrappers/autoencoder.py

import torch
from torch import nn
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.model_registry import MODEL_REGISTRY
from refrakt_core.registry.wrapper_registry import register_wrapper

@register_wrapper("autoencoder")
class AutoencoderWrapper(nn.Module):
    def __init__(self, model: nn.Module, variant: str = "simple"):
        super().__init__()
        self.backbone = model
        self.variant = variant

    def forward(self, x):
        output = self.backbone(x)

        if self.variant == "vae":
            return ModelOutput(
                reconstruction=output["recon"],
                extra={
                    "mu": output["mu"],
                    "logvar": output["logvar"],
                }
            )
        elif self.variant == "mae":
            return ModelOutput(
                reconstruction=output["recon"],
                extra={
                    "mask": output["mask"],
                    "original_patches": output["original_patches"],
                }
            )
        else:
            return ModelOutput(reconstruction=output)

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).reconstruction
