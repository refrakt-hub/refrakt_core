from typing import Any, Optional

import torch

from refrakt_core.losses.ntxent import NTXentLoss
from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.schema.loss_output import LossOutput


@register_loss("ntxent_wrapped", mode="embedding")
class NTXentLossWrapper(BaseLoss):
    def __init__(
        self, temperature: float = 0.5, name: Optional[str] = None, **kwargs: Any
    ) -> None:
        super().__init__(name=name)
        self.temperature = temperature
        self.inner_loss = NTXentLoss(temperature=temperature, name=name)
        self.config = kwargs

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> LossOutput:
        loss = self.inner_loss(z1, z2)
        return LossOutput(
            total=loss,
            components={"ntxent": loss},
        )
