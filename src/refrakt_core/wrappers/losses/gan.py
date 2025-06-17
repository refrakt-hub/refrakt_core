# wrappers/gan.py
from typing import Dict, Optional
import torch
from torch import nn

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.losses.gan import GANLoss

@register_loss("gan", mode="logits")
class GANLossWrapper(nn.Module):
    def __init__(self, loss_params: Optional[Dict] = None):
        super().__init__()
        loss_params = loss_params or {}
        self.loss_fn = GANLoss(**loss_params)
        self.required_fields = ["pred", "target_is_real"]

    def forward(self, output: ModelOutput, target=None) -> LossOutput:
        pred = output.logits
        target_is_real = output.extra.get("target_is_real")
        
        if pred is None or target_is_real is None:
            missing = [f for f in self.required_fields if locals()[f] is None]
            raise ValueError(f"Missing required fields: {missing}")

        loss = self.loss_fn(pred, target_is_real)
        return LossOutput(total=loss, components={"gan": loss})