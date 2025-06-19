import torch
from torch import nn
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.wrapper_registry import register_wrapper

@register_wrapper("resnet")
class ResNetWrapper(nn.Module):
    """
    Wrapper around a pre-initialized ResNet model to standardize its output.
    
    Args:
        model (nn.Module): A ResNet model instance (e.g., ResNet18, ResNet50).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model

    def forward(self, x):
        logits = self.backbone(x)
        return ModelOutput(logits=logits)

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).logits
