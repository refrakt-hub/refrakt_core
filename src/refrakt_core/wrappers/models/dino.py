from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch import nn

from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("dino")
class DINOWrapper(nn.Module):
    def __init__(self, model: Any, **kwargs: Any):
        super().__init__()

        from refrakt_core.models.dino import DINOModelWrapper

        if isinstance(model, (dict, DictConfig)):
            # Extract known model keys
            backbone = model.get("backbone", "resnet18")
            out_dim = model.get("out_dim", 65536)

            # Create the DINO model with only the parameters it expects
            self.dino_model = DINOModelWrapper(backbone=backbone, out_dim=out_dim)
        elif isinstance(model, nn.Module):
            self.dino_model = model
        else:
            raise TypeError(f"[DINOWrapper] Invalid model type: {type(model)}")

        # Store wrapper config, filtering out model initialization parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in {"backbone", "out_dim", "model"}
        }
        self.wrapper_config = {"wrapper_type": "dino", **filtered_kwargs}

    def forward(self, x: torch.Tensor, teacher: bool = False, **kwargs) -> ModelOutput:
        # Filter out any unexpected kwargs that might be passed from the training loop
        # Only pass the arguments that the DINO model's forward method expects
        valid_forward_args = {"teacher": teacher}

        # Forward to the DINO model with only valid arguments
        embeddings = self.dino_model(x, **valid_forward_args)
        output = ModelOutput(embeddings=embeddings, loss_components={})

        # Add attention maps if available
        if hasattr(self.dino_model, "backbone") and hasattr(
            self.dino_model.backbone, "get_attention_maps"
        ):
            output.attention_maps = self.dino_model.backbone.get_attention_maps(x)

        output.extra["wrapper_config"] = self.wrapper_config
        return output

    def update_teacher(self):
        if hasattr(self.dino_model, "update_teacher"):
            return self.dino_model.update_teacher()
        raise AttributeError(
            "[DINOWrapper] Inner model has no method 'update_teacher()'"
        )

    def parameters(self, recurse: bool = True):
        return self.dino_model.student_head.parameters()

    def named_parameters(self, prefix="", recurse=True):
        return self.dino_model.student_head.named_parameters(
            prefix=prefix, recurse=recurse
        )
