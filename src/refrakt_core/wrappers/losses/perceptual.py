# wrappers/perceptual.py
from typing import Any, Dict, Optional

from torch import nn

from refrakt_core.losses.perceptual import PerceptualLoss
from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


@register_loss("perceptual", mode="features")
class PerceptualLossWrapper(nn.Module):
    def __init__(self, loss_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        loss_params = loss_params or {}
        self.loss_fn = PerceptualLoss(**loss_params)
        self.required_fields = ["sr", "hr"]

    def forward(self, output: ModelOutput, target: Any = None) -> LossOutput:
        sr = output.reconstruction
        hr = target  # Use target for high-res images

        if sr is None or hr is None:
            missing = []
            if sr is None:
                missing.append("sr")
            if hr is None:
                missing.append("hr")
            raise ValueError(f"Missing required fields: {missing}")

        loss = self.loss_fn(sr, hr)
        return LossOutput(total=loss, components={"perceptual": loss})
