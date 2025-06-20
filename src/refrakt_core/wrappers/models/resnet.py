import torch
from torch import nn
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.wrapper_registry import register_wrapper

@register_wrapper("resnet")
class ResNetWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.head = model.fc

    def forward(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        logits = self.head(feats)

        return ModelOutput(logits=logits, embeddings=feats)

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).logits
