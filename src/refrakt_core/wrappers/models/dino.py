from typing import Any, Iterator

import torch
from omegaconf import DictConfig
from torch import nn

from refrakt_core.models.dino import DINOModelWrapper
from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("dino")
class DINOWrapper(nn.Module):
    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__()
        self.dino_model: DINOModelWrapper
        if isinstance(model, (dict, DictConfig)):
            backbone = model.get("backbone", "resnet18")
            out_dim = model.get("out_dim", 65536)
            self.dino_model = DINOModelWrapper(backbone=backbone, out_dim=out_dim)
        elif isinstance(model, nn.Module):
            self.dino_model = model  # type: ignore
        else:
            raise TypeError(f"[DINOWrapper] Invalid model type: {type(model)}")

        # Store wrapper config, filtering out model initialization parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in {"backbone", "out_dim", "model"}
        }
        self.wrapper_config = {"wrapper_type": "dino", **filtered_kwargs}

    def forward(
        self, x: torch.Tensor, teacher: bool = False, **kwargs: Any
    ) -> ModelOutput:
        # Filter out any unexpected kwargs that might be passed from the training loop
        # Only pass the arguments that the DINO model's forward method expects
        valid_forward_args = {"teacher": teacher}

        embeddings = self.dino_model(x, **valid_forward_args)  # type: ignore
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor from dino_model forward, got {type(embeddings)}"
            )
        output = ModelOutput(embeddings=embeddings, loss_components={})

        # Add attention maps if available
        if hasattr(self.dino_model, "backbone") and hasattr(
            self.dino_model.backbone, "get_attention_maps"
        ):
            output.attention_maps = self.dino_model.backbone.get_attention_maps(x)

        output.extra["wrapper_config"] = self.wrapper_config
        return output

    def update_teacher(self) -> None:
        if hasattr(self.dino_model, "update_teacher"):
            return self.dino_model.update_teacher()
        raise AttributeError(
            "[DINOWrapper] Inner model has no method 'update_teacher()'"
        )

    def parameters(self, recurse: bool = True) -> Any:
        return self.dino_model.student_head.parameters()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Any]:
        return self.dino_model.student_head.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )
