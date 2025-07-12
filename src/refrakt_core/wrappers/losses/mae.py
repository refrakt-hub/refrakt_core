from typing import Any

from torch import nn

from refrakt_core.losses.mae import MAELoss
from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


@register_loss("mae_wrapped", mode="reconstruction")
class MAELossWrapper(nn.Module):
    def __init__(self, **loss_params: Any) -> None:
        super().__init__()
        self.loss_fn = MAELoss(**(loss_params or {}))

    def forward(self, output: ModelOutput, target: Any = None) -> LossOutput:
        recon = output.reconstruction
        mask = output.extra.get("mask")
        patches = output.extra.get("original_patches")

        if None in (recon, mask, patches):
            raise ValueError(
                f"Missing required fields for MAE loss: "
                f"{['reconstruction', 'mask', 'original_patches']}"
            )

        loss = self.loss_fn({"recon": recon, "mask": mask, "original_patches": patches})
        return LossOutput(total=loss, components={"masked_mse": loss})
