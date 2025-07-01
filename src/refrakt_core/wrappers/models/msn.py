"""
Wrapper for the MSN model that returns structured ModelOutput.
"""

import torch
from torch import nn
from typing import Dict
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.wrapper_registry import register_wrapper


@register_wrapper("msn")
class MSNWrapper(nn.Module):
    """
    MSN Wrapper to return structured ModelOutput for training and logging.

    Args:
        model: Instance of MSNModel
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Dict[str, torch.Tensor]) -> ModelOutput:
        """
        Args:
            x: Dictionary containing 'anchor' and 'target' inputs
        
        Returns:
            ModelOutput with embeddings, extra fields, and targets
        """
        # Debug: check input shapes
        x_anchor = x['anchor']
        x_target = x['target']
        assert isinstance(x_anchor, torch.Tensor), f"x['anchor'] is not a tensor: {type(x_anchor)}"
        assert isinstance(x_target, torch.Tensor), f"x['target'] is not a tensor: {type(x_target)}"
        assert x_anchor.ndim == 4, f"x['anchor'] shape: {x_anchor.shape} (expected 4D tensor)"
        assert x_target.ndim == 4, f"x['target'] shape: {x_target.shape} (expected 4D tensor)"

        # Unpack inputs
        x_anchor = x['anchor']
        x_target = x['target']

        # Forward pass through MSNModel
        z_anchor, z_target, prototypes = self.model(x_anchor, x_target)
        
        return ModelOutput(
            embeddings=z_anchor,
            targets=x_target,
            extra={
                "z_target": z_target,
                "prototypes": prototypes
            }
        )