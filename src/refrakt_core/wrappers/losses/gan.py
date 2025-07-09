from typing import Any, Dict, Optional

from refrakt_core.losses.gan import GANLoss
from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


@register_loss("gan_wrapped", mode="logits")
class GANLossWrapper(BaseLoss):
    def __init__(self, loss_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        loss_params = loss_params or {}
        self.loss_fn = GANLoss(**loss_params)
        self.required_fields = ["logits", "target_is_real"]

    def forward(self, output: ModelOutput, target: Any = None) -> LossOutput:
        logits = output.logits
        target_is_real = output.extra.get("target_is_real")

        if logits is None or target_is_real is None:
            missing = [f for f in self.required_fields if f not in output.extra]
            raise ValueError(f"Missing required fields: {missing}")
        loss = self.loss_fn(logits, target_is_real)
        return LossOutput(total=loss, components={"gan": loss})
