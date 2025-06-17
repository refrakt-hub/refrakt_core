"""
The ResNet wrapper that dyynamically loads a ResNet model from the registry
"""

import torch
from torch import nn
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.model_registry import MODEL_REGISTRY
from refrakt_core.registry.wrapper_registry import register_wrapper

@register_wrapper("resnet")
class ResNetWrapper(nn.Module):
    """
    Wrapper around ResNet model that returns standardized ModelOutput.
    
    Args:
        model_name (str): Name of the registered model (e.g. 'resnet18', 'resnet50').
        model_params (dict): Parameters for the base model (e.g. in_channels, num_classes).
    """

    def __init__(self, model_name: str, model_params: dict):
        super().__init__()
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"[ResNetWrapper] Model '{model_name}' not registered in MODEL_REGISTRY.")

        self.backbone = MODEL_REGISTRY[model_name](**model_params)

    def forward(self, x):
        logits = self.backbone(x)
        return ModelOutput(logits=logits)

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        """Used only for graph logging; returns just the raw output tensor."""
        return self.forward(x).logits  # or `.reconstruction`, etc. depending on task
