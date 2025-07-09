# wrappers/dino.py

import torch

from refrakt_core.losses.dino import DINOLoss
from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.schema.loss_output import LossOutput

__all__ = ["DINOLoss", "DINOLossWrapper"]


@register_loss("dino_wrapped", mode="embedding")
class DINOLossWrapper(BaseLoss):
    def __init__(
        self,
        out_dim: int = 1024,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__(name="DINOLossWrapper")
        from .dino import DINOLoss  # Import original loss

        self.loss_fn = DINOLoss(
            out_dim=out_dim,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )

    def forward(
        self, student_out: torch.Tensor, teacher_out: torch.Tensor
    ) -> LossOutput:
        loss = self.loss_fn(student_out, teacher_out)
        return LossOutput(total=loss, components={"dino": loss})
