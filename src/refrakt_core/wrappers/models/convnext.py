"""
The ConvNeXt wrapper that dynamically loads the ConvNeXt model from the registry.
"""

import torch
from torch import nn

from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("convnext")
class ConvNeXtWrapper(nn.Module):
    """
    Wrapper around ConvNeXt model that returns standardized ModelOutput.

    Args:
        model_name (str): Name of the registered model (e.g. 'convnext').
        model_params (dict): Parameters for the base model (e.g. in_channels, num_classes).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass for ConvNeXtWrapper.
        Returns ModelOutput with logits and embeddings.
        """
        embeddings = self.backbone(x, return_features=True)
        logits = self.backbone.fc(embeddings)
        return ModelOutput(logits=logits, embeddings=embeddings)

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lightweight forward pass for graph tracing / visualization.

        Returns:
            torch.Tensor: Only the logits.
        """
        out = self.forward(x).logits
        if isinstance(out, torch.Tensor):
            return out
        raise TypeError(f"Expected torch.Tensor for logits, got {type(out)}")
