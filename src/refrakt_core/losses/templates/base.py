import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name or self.__class__.__name__

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def freeze(self):
        """Freeze all parameters in this loss module."""
        for param in self.parameters():
            param.requires_grad = False

    def requires_grad(self, mode=True):
        """Set requires_grad for all parameters."""
        for param in self.parameters():
            param.requires_grad = mode
        return self

    def log(self):
        """Return a basic config dictionary for logging purposes."""
        return {
            "loss_name": self.name,
            "trainable": any(p.requires_grad for p in self.parameters()),
        }

    def get_config(self):
        """Override this to return detailed config of the loss."""
        return {
            "name": self.name,
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "total_params": sum(p.numel() for p in self.parameters()),
        }

    def extra_repr(self):
        return f"name={self.name}"
