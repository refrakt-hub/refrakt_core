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

    def __init__(self, model_name: str = "base_model", model_type: str = "generic") -> None:
        """
        Initialize the base model.

        Args:
            model_name (str): Name identifier for the model. Defaults to "base_model".
            model_type (str): Type/architecture of the model. Defaults to "generic".
        """
        super().__init__()
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                return torch.softmax(output, dim=1) if kwargs.get("return_probs", False) else torch.argmax(output, dim=1)
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
        return super().to(device)  # type: ignore[return-value]
