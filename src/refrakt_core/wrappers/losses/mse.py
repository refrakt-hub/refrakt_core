from typing import Any

import torch
from torch import nn

from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


@register_loss("mse_wrapped", mode="reconstruction")
class MSELossWrapper(nn.Module):
    def __init__(self, **loss_params: Any) -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss(**(loss_params or {}))

    def forward(self, output: ModelOutput, target: Any = None) -> LossOutput:
        recon = output.reconstruction

        if recon is None:
            raise ValueError("Missing reconstruction field in ModelOutput")

        # Use target if provided, otherwise use the reconstruction itself
        effective_target = target if target is not None else recon

        loss = self.loss_fn(recon, effective_target)
        return LossOutput(total=loss, components={"mse": loss}) 