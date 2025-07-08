from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from refrakt_core.schema.model_output import ModelOutput


class DefaultModelWrapper(nn.Module):
    """
    Fallback model wrapper that standardizes output to ModelOutput.
    """

    def __init__(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        modules: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if modules is None or "get_model" not in modules:
            raise ValueError(
                "modules['get_model'] must be provided for DefaultModelWrapper."
            )
        self.model = modules["get_model"](model_name, **model_params)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> ModelOutput:
        output = self.model(x, **kwargs)

        if isinstance(output, ModelOutput):
            return output

        if isinstance(output, torch.Tensor):
            return ModelOutput(logits=output)

        if isinstance(output, dict):
            return ModelOutput(
                logits=output.get("logits"),
                embeddings=output.get("embeddings"),
                reconstruction=output.get("reconstruction"),
                attention_maps=output.get("attention_maps"),
                image=output.get("image"),
                loss_components=output.get("loss_components", {}),
                extra={
                    k: v
                    for k, v in output.items()
                    if k
                    not in {
                        "logits",
                        "embeddings",
                        "reconstruction",
                        "attention_maps",
                        "image",
                        "loss_components",
                    }
                },
            )

        raise ValueError(f"Unsupported output type from model: {type(output)}")

    def parameters(self, recurse: bool = True) -> Any:
        return self.model.parameters(recurse=recurse)
