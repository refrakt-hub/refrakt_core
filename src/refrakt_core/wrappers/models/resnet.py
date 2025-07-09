import torch
from torch import nn

from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("resnet")
class ResNetWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        # Keep the original model as is
        self.backbone = model

    def forward(self, x: torch.Tensor) -> ModelOutput:
        feats = self.backbone(x, return_features=True)
        logits = self.backbone.fc(feats)
        return ModelOutput(logits=logits, embeddings=feats)

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, torch.Tensor):
            return out
        raise TypeError(
            "Expected torch.Tensor from backbone forward, got {}".format(type(out))
        )
