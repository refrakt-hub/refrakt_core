# wrappers/dino.py
from typing import Dict, Optional
import torch
from torch import nn

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.losses.dino import DINOLoss

@register_loss("dino", mode="embedding")
class DINOLossWrapper(nn.Module):
    def __init__(self, loss_params: Optional[Dict] = None):
        super().__init__()
        loss_params = loss_params or {}
        self.loss_fn = DINOLoss(**loss_params)
        self.required_fields = ["student_output", "teacher_output"]

    def forward(self, output: ModelOutput, target=None) -> LossOutput:
        student = output.embeddings
        teacher = output.extra.get("teacher_output")
        
        if student is None or teacher is None:
            missing = [f for f in self.required_fields if locals()[f] is None]
            raise ValueError(f"Missing required fields: {missing}")

        loss = self.loss_fn(student, teacher)
        return LossOutput(total=loss, components={"dino": loss})