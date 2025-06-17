# wrappers/nt_xent.py
from typing import Dict, Optional
import torch
from torch import nn

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.losses.ntxent import NTXentLoss

@register_loss("nt_xent", mode="embedding")
class NTXentLossWrapper(nn.Module):
    def __init__(self, loss_params: Optional[Dict] = None):
        super().__init__()
        loss_params = loss_params or {}
        self.loss_fn = NTXentLoss(**loss_params)
        self.required_fields = ["z1", "z2"]

    def forward(self, output: ModelOutput, target=None) -> LossOutput:
        z1 = output.embeddings
        z2 = output.extra.get("z2")
        
        if z1 is None or z2 is None:
            missing = [f for f in self.required_fields if locals()[f] is None]
            raise ValueError(f"Missing required fields: {missing}")

        loss = self.loss_fn(z1, z2)
        return LossOutput(total=loss, components={"nt_xent": loss})