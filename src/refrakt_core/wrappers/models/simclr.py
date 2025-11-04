from typing import Any

import torch

from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("simclr")
class SimCLRWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.model = model
        self.wrapper_config = kwargs

        # Set model attributes for XAI detection
        self.model_name = "simclr"
        self.model_type = "contrastive"

    def forward(self, x: torch.Tensor) -> ModelOutput:
        embeddings = self.model(x)
        return ModelOutput(
            embeddings=embeddings,
            extra={"wrapper_type": "simclr", **self.wrapper_config},
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "encode"):
            return self.model.encode(x)
        return self.model.forward(x)
