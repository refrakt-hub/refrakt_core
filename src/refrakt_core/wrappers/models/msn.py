"""
Wrapper for the MSN model that returns structured ModelOutput.
"""

from typing import Dict

import torch
from torch import nn

from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


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
        # Unpack inputs
        x_anchor = x["anchor"]
        x_target = x["target"]

        # Forward pass through MSNModel
        z_anchor, z_target, prototypes = self.model(x_anchor, x_target)

        return ModelOutput(
            embeddings=z_anchor,
            targets=x_target,
            extra={"z_target": z_target, "prototypes": prototypes},
        )
