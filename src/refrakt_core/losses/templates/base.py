from typing import Any, Optional

import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.name = name or self.__class__.__name__

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement the forward method.")

    def freeze(self) -> None:
        """Freeze all parameters in this loss module."""
        for param in self.parameters():
            param.requires_grad = False

    def requires_grad(self, mode: bool = True) -> "BaseLoss":
        """Set requires_grad for all parameters."""
        for param in self.parameters():
            param.requires_grad = mode
        return self

    def log(self) -> dict[str, Any]:
        """Return a basic config dictionary for logging purposes."""
        return {
            "loss_name": self.name,
            "trainable": any(p.requires_grad for p in self.parameters()),
        }

    def get_config(self) -> dict[str, Any]:
        """Override this to return detailed config of the loss."""
        return {
            "name": self.name,
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "total_params": sum(p.numel() for p in self.parameters()),
        }

    def extra_repr(self) -> str:
        return f"name={self.name}"
