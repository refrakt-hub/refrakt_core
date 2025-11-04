from typing import Any

import torch
from torch import nn

from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("swin")
class SwinTransformerWrapper(nn.Module):
    """
    Wrapper for the Swin Transformer model.

    Returns standardized ModelOutput containing logits and embeddings.
    """

    def __init__(self, model: nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.backbone = model
        self.layer_metrics = {}
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.backbone.named_modules():
            if len(list(module.children())) == 0:
                self.layer_metrics[name] = {"activation_mean": 0.0, "grad_mean": 0.0}
                module.register_forward_hook(self._make_activation_hook(name))
                module.register_full_backward_hook(self._make_grad_hook(name))

    def _make_activation_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.layer_metrics[name]["activation_mean"] = (
                    output.detach().mean().item()
                )

        return hook

    def _make_grad_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output and isinstance(grad_output[0], torch.Tensor):
                self.layer_metrics[name]["grad_mean"] = (
                    grad_output[0].detach().mean().item()
                )

        return hook

    def get_layer_metrics(self):
        return self.layer_metrics.copy()

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Args:
            x (Tensor): Input image batch of shape (B, C, H, W)

        Returns:
            ModelOutput:
                - logits: final classification outputs (B, num_classes)
                - embeddings: pre-logit pooled features (B, hidden_dim)
        """
        # Replicating forward logic to intercept pre-logit embedding
        x = self.backbone.embedding(x)
        x = self.backbone.patch1(self.backbone.stage1(x))
        x = self.backbone.patch2(self.backbone.stage2(x))
        x = self.backbone.stage3_1(x)
        x = self.backbone.stage3_2(x)
        x = self.backbone.stage3_3(x)
        x = self.backbone.patch3(x)
        x = self.backbone.stage4(x)

        embeddings = x.mean(dim=1)  # global average over patch tokens
        logits = self.backbone.head(embeddings)

        return ModelOutput(logits=logits, embeddings=embeddings)

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lightweight forward pass for graph tracing / visualization.

        Returns:
            torch.Tensor: Only the logits.
        """
        logits = self.forward(x).logits
        return (
            torch.as_tensor(logits) if not isinstance(logits, torch.Tensor) else logits
        )

    def __call__(self, *args, **kwargs):
        if getattr(self, "_captum_tracing", False):
            return self.forward_for_graph(*args, **kwargs)
        return super().__call__(*args, **kwargs)
