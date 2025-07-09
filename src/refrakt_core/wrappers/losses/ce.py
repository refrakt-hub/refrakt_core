"""
Wrapper for CrossEntropyLoss using register_loss with mode='logits'.
"""

from typing import Any, Dict, Optional

from torch import Tensor, nn

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


@register_loss("ce_wrapped", mode="logits")
class CrossEntropyLossWrapper(BaseLoss):
    """
    Wrapper for CrossEntropyLoss that uses ModelOutput and returns LossOutput.

    Args:
        weight (Optional[Tensor]): Class weights.
        label_smoothing (float): Label smoothing factor.
        device (str): Device for computation.
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(name="CrossEntropyLoss")
        self.device = device
        self.label_smoothing = label_smoothing
        self.weight = weight.to(device) if weight is not None else None

        self.loss_fn = nn.CrossEntropyLoss(
            weight=self.weight, label_smoothing=self.label_smoothing
        )

    def forward(self, output: ModelOutput, target: Tensor) -> LossOutput:
        """
        Args:
            output (ModelOutput): Should contain `.logits` (N, C).
            target (Tensor): Ground-truth labels of shape (N,).

        Returns:
            LossOutput: Contains total loss and its components.
        """
        if output.logits is None:
            raise ValueError("[CrossEntropyLossWrapper] Missing logits in ModelOutput.")
        if target is None:
            raise ValueError("[CrossEntropyLossWrapper] Target is required.")

        loss = self.loss_fn(output.logits, target)

        return LossOutput(total=loss, components={"cross_entropy": loss})

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "label_smoothing": self.label_smoothing,
            "device": self.device,
        }

    def extra_repr(self) -> str:
        return f"name={self.name}, label_smoothing={self.label_smoothing}, device={self.device}"
