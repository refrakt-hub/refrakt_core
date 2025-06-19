"""
The ConvNeXt wrapper that dynamically loads the ConvNeXt model from the registry.
"""

from torch import nn
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.model_registry import MODEL_REGISTRY
from refrakt_core.registry.wrapper_registry import register_wrapper

@register_wrapper("convnext")
class ConvNeXtWrapper(nn.Module):
    """
    Wrapper around ConvNeXt model that returns standardized ModelOutput.
    
    Args:
        model_name (str): Name of the registered model (e.g. 'convnext').
        model_params (dict): Parameters for the base model (e.g. in_channels, num_classes).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model

    def forward(self, x):
        logits = self.backbone(x)
        return ModelOutput(logits=logits)
