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

    def forward(self, x: torch.Tensor) -> ModelOutput:
        embeddings = self.model(x)
        return ModelOutput(
            embeddings=embeddings,
            extra={"wrapper_type": "simclr", **self.wrapper_config},
        )
