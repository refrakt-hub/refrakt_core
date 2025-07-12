"""
Wrapper for the MSN model that returns structured ModelOutput.
"""

from typing import Any, Dict

import torch
from omegaconf import DictConfig
from torch import nn

from refrakt_core.models.msn import MSNModel
from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("msn")
class MSNWrapper(nn.Module):
    """
    MSN Wrapper to return structured ModelOutput for training and logging.

    Args:
        model: Either an instance of MSNModel or a configuration dict
    """

    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__()
        
        if isinstance(model, (dict, DictConfig)):
            # Extract MSN model parameters from config
            encoder_name = model.get("encoder_name", "resnet18")
            projector_dim = model.get("projector_dim", 256)
            num_prototypes = model.get("num_prototypes", 1024)
            pretrained = model.get("pretrained", True)
            
            self.msn_model = MSNModel(
                encoder_name=encoder_name,
                projector_dim=projector_dim,
                num_prototypes=num_prototypes,
                pretrained=pretrained,
            )
        elif isinstance(model, nn.Module):
            self.msn_model = model
        else:
            raise TypeError(f"[MSNWrapper] Invalid model type: {type(model)}")

        # Store wrapper config, filtering out model initialization parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in {"encoder_name", "projector_dim", "num_prototypes", "pretrained", "model"}
        }
        self.wrapper_config = {"wrapper_type": "msn", **filtered_kwargs}

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
        z_anchor, z_target, prototypes = self.msn_model(x_anchor, x_target)

        return ModelOutput(
            embeddings=z_anchor,
            targets=x_target,
            extra={"z_target": z_target, "prototypes": prototypes},
        )
