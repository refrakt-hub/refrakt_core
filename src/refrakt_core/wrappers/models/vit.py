from typing import Any

import torch
from torch import nn

from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("vit")
class ViTWrapper(nn.Module):
    """
    Wrapper around a Vision Transformer (ViT) model for classification and feature extraction.

    Args:
        model (nn.Module): VisionTransformer instance.
        **kwargs: Additional arguments for compatibility.
    """

    def __init__(self, model: nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.backbone = model

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass through ViT.

        Returns:
            ModelOutput: Contains logits and optional CLS embeddings.
        """
        cls_token = self.backbone.forward_features(x)
        logits = self.backbone.mlp_head(cls_token)

        return ModelOutput(
            logits=logits,
            embeddings=cls_token,  # useful for contrastive or similarity learning
        )

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for visualizing computation graph (e.g., in W&B, TensorBoard).

        Returns:
            torch.Tensor: Logits only.
        """
        logits = self.forward(x).logits
        return (
            torch.as_tensor(logits) if not isinstance(logits, torch.Tensor) else logits
        )
