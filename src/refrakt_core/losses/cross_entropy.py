"""
Cross-Entropy loss with optional label smoothing, commonly used for classification tasks.
"""

from typing import Dict, Optional

from torch import Tensor, nn

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss


@register_loss("cross_entropy")
class CrossEntropyLoss(BaseLoss):
    """
    Cross-Entropy Loss with optional label smoothing.

    Args:
        weight (Optional[Tensor]): Class weights tensor of shape (C,) or None.
        label_smoothing (float): Smoothing factor for label smoothing (default: 0.0).
        device (str): Device to use for tensors (default: "cuda").
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(name="CrossEntropyLoss")
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.device = device
        self.loss = nn.CrossEntropyLoss(
            weight=weight.to(device) if weight is not None else None,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the cross-entropy loss.

        Args:
            pred (Tensor): Predictions of shape (N, C).
            target (Tensor): Ground-truth labels of shape (N,).

        Returns:
            Tensor: Scalar loss value.

        Raises:
            ValueError: If tensor shapes are incompatible.
        """
        if pred.ndim != 2:
            raise ValueError(f"Expected pred shape (N, C), got {pred.shape}")
        if target.ndim != 1:
            raise ValueError(f"Expected target shape (N,), got {target.shape}")
        if pred.size(0) != target.size(0):
            raise ValueError(
                f"Batch size mismatch: pred={pred.size(0)}, target={target.size(0)}"
            )

        return self.loss(pred, target)  # type: ignore[no-any-return]

    def get_config(self) -> Dict[str, object]:
        """
        Return configuration of the loss function.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            **super().get_config(),
            "weight": self.weight,
            "label_smoothing": self.label_smoothing,
            "device": self.device,
        }

    def extra_repr(self) -> str:
        return f"name={self.name}, label_smoothing={self.label_smoothing}, device={self.device}"
