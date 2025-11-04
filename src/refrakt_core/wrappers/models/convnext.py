"""
The ConvNeXt wrapper that dynamically loads the ConvNeXt model from the registry.
"""

import torch
from torch import nn

from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("convnext")
class ConvNeXtWrapper(nn.Module):
    """
    Wrapper around ConvNeXt model that returns standardized ModelOutput.

    Args:
        model_name (str): Name of the registered model (e.g. 'convnext').
        model_params (dict): Parameters for the base model (e.g. in_channels, num_classes).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.fc = model.fc
        self.layer_metrics = {}
        self._register_hooks()

    def __call__(self, *args, **kwargs):
        if getattr(self, "_captum_tracing", False):
            return self.forward_for_graph(*args, **kwargs)
        return super().__call__(*args, **kwargs)

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
        Forward pass for ConvNeXtWrapper.
        Returns ModelOutput with logits and embeddings.
        """
        embeddings = self.backbone(x)
        embeddings = embeddings.view(embeddings.size(0), -1)
        logits = self.fc(embeddings)
        return ModelOutput(logits=logits, embeddings=embeddings)

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lightweight forward pass for graph tracing / visualization.

        Returns:
            torch.Tensor: Only the logits.
        """
        out = self.forward(x).logits
        if isinstance(out, torch.Tensor):
            return out
        raise TypeError(f"Expected torch.Tensor for logits, got {type(out)}")
