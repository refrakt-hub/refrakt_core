# wrappers/autoencoder.py

import torch
from torch import nn
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.model_registry import MODEL_REGISTRY
from refrakt_core.registry.wrapper_registry import register_wrapper


@register_wrapper("autoencoder")
class AutoencoderWrapper(nn.Module):
    def __init__(self, model_name: str, model_params: dict):
        super().__init__()
        self.variant = model_params.get("variant") or model_params.get("type") or "simple"

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"[AutoencoderWrapper] Model '{model_name}' not found in registry.")
        self.backbone = MODEL_REGISTRY[model_name](**model_params)

    def forward(self, x):
        output = self.backbone(x)
        model_out = None
        
        if self.variant == "vae":
            
            model_out = ModelOutput(
                reconstruction=output["recon"],
                extra={
                    "mu": output["mu"],
                    "logvar": output["logvar"],
                }
            )

        elif self.variant == "mae":
            model_out = ModelOutput(
                reconstruction=output["recon"],
                extra={
                    "mask": output["mask"],
                    "original_patches": output["original_patches"],
                }
            )
        else:
            model_out = ModelOutput(reconstruction=output)

        if model_out.reconstruction is None:
            raise RuntimeError("[AutoencoderWrapper] reconstruction is None in ModelOutput!")

        return model_out
    
    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).reconstruction
