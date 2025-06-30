import torch
from torch import nn
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.wrapper_registry import register_wrapper

@register_wrapper("resnet")
class ResNetWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # Keep the original model as is
        self.backbone = model

    def forward(self, x):
        # Get features using the backbone's forward with return_features=True
        feats = self.backbone(x, return_features=True)
        # Get logits by applying fc layer to features
        logits = self.backbone.fc(feats)
        return ModelOutput(logits=logits, embeddings=feats)

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        # For tracing, directly use the backbone's forward method
        # This avoids the wrapper complexity and ensures proper tensor shapes
        return self.backbone(x)
