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

    def __call__(self, *args, **kwargs):
        if getattr(self, "_captum_tracing", False):
            return self.forward_for_graph(*args, **kwargs)
        return super().__call__(*args, **kwargs)
