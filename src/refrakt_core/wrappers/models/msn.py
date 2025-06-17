"""
Wrapper for the MSN model that returns structured ModelOutput.
"""

from torch import nn
from typing import Dict, Any, Tuple
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.model_registry import MODEL_REGISTRY
from refrakt_core.registry.wrapper_registry import register_wrapper


@register_wrapper("msn")
class MSNWrapper(nn.Module):
    """
    MSN Wrapper to return structured ModelOutput for training and logging.

    Args:
        model_name (str): Should be 'msn'.
        model_params (dict): Parameters for MSNModel (encoder_name, projector_dim, etc.)
    """

    def __init__(self, model_name: str, model_params: Dict[str, Any]):
        super().__init__()
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"[MSNWrapper] Model '{model_name}' not registered.")
        self.backbone = MODEL_REGISTRY[model_name](**model_params)

    def forward(self, x_anchor, x_target) -> ModelOutput:
        z_anchor, z_target, prototypes = self.backbone(x_anchor, x_target)
        return ModelOutput(
            embeddings=z_anchor,
            loss_components={
                "z_target": z_target,
                "prototypes": prototypes,
            },
            extra={
                "prototypes": prototypes.detach().cpu(),
                "z_target": z_target.detach().cpu()
            }
        )
