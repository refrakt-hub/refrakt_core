# refrakt_core/losses/cross_entropy.py

from typing import Dict, Optional
import torch
from torch import Tensor, nn

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.registry.loss_registry import register_loss

@register_loss("cross_entropy", mode="logits")
class CrossEntropyLoss(BaseLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        device: str = "cuda"
    ) -> None:
        super().__init__(name="CrossEntropyLoss")
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.device = device
        self.loss = nn.CrossEntropyLoss(
            weight=weight.to(device) if weight is not None else None,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: Tensor, target: Tensor) -> LossOutput:
        if logits.ndim != 2 or target.ndim != 1:
            raise ValueError(f"Incompatible shapes: logits={logits.shape}, target={target.shape}")
        if logits.size(0) != target.size(0):
            raise ValueError(f"Batch mismatch: logits={logits.size(0)} vs target={target.size(0)}")

        loss = self.loss(logits, target)
        return LossOutput(total=loss, components={"cross_entropy": loss})

    def get_config(self) -> Dict[str, Optional[float]]:
        return {
            **super().get_config(),
            "weight": self.weight,
            "label_smoothing": self.label_smoothing,
            "device": self.device,
        }

    def extra_repr(self) -> str:
        return f"name={self.name}, label_smoothing={self.label_smoothing}, device={self.device}"
