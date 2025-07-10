"""Abstract base model definition for neural network architectures."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all neural network models.

    This class provides a common interface for different model architectures,
    including methods for forward pass, prediction, and saving/loading model weights.
    """

    def __init__(
        self, model_name: str = "base_model", model_type: str = "generic"
    ) -> None:
        """
        Initialize the base model.

        Args:
            model_name (str): Name identifier for the model. Defaults to "base_model".
            model_type (str): Type/architecture of the model. Defaults to "generic".
        """
        super().__init__()
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_name: str = model_name
        self.model_type: str = model_type

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass

    def predict(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Perform prediction with the model.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional arguments (e.g., return_probs for classifiers).

        Returns:
            torch.Tensor: Model predictions.
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device) if x.device != self.device else x
            output = self.forward(x)

            if self.model_type == "classifier":
                return (
                    torch.softmax(output, dim=1)
                    if kwargs.get("return_probs", False)
                    else torch.argmax(output, dim=1)
                )
            if self.model_type == "autoencoder":
                return output
            return output

    def save_model(self, path: str) -> None:
        """
        Save model weights to disk.

        Args:
            path (str): Path to save the model.
        """
        model_state: Dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "model_type": self.model_type,
        }
        torch.save(model_state, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model weights from disk.

        Args:
            path (str): Path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.model_name = checkpoint.get("model_name", self.model_name)
        self.model_type = checkpoint.get("model_type", self.model_type)
        print(f"Model loaded from {path}")

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.

        Returns:
            Dict[str, Any]: Model summary information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

    def to_device(self, device: torch.device) -> "BaseModel":
        """
        Move model to specified device.

        Args:
            device (torch.device): Device to move the model to.

        Returns:
            BaseModel: Self reference for method chaining.
        """
        self.device = device
        return super().to(device)

    def dummy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optional: Override this in models that need special input logic for tracing/graph logging.
        Default: Assumes single input is sufficient (for simple models).
        """
        return self.forward(x)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Handles both normal forward calls and special cases like model graph tracing.
        If only a single tensor is passed during a tracing context (e.g., W&B or TorchScript),
        we route it through dummy_forward.
        """
        if (
            len(args) == 1
            and isinstance(args[0], torch.Tensor)
            and self._is_graph_tracing()
        ):
            return self.dummy_forward(args[0])
        return super().__call__(*args, **kwargs)

    def _is_graph_tracing(self) -> bool:
        """
        Detects whether we're in a model graph tracing context like wandb.watch or torch.jit.
        """
        import inspect

        for frame in inspect.stack():
            if any(
                keyword in frame.filename.lower() for keyword in ("wandb", "torch/jit")
            ):
                return True
        return False

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, value: torch.device) -> None:
        self._device = value
