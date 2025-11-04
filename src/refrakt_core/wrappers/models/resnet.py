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
        self.layer_metrics = {}
        self._register_hooks()

    def _register_hooks(self):
        # Register hooks for all modules except the top-level wrapper
        for name, module in self.backbone.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
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

    def __call__(self, *args, **kwargs):
        # Captum workaround: if _captum_tracing is set, return tensor for XAI
        if getattr(self, "_captum_tracing", False):
            return self.forward_for_graph(*args, **kwargs)
        return super().__call__(*args, **kwargs)
