"""
MSNLossWrapper: A wrapper class for the Masked Siamese Network (MSN) loss.
"""

from typing import Any

import torch
from torch import nn

from refrakt_core.losses.msn import MSNLoss
from refrakt_core.registry.loss_registry import register_loss
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


@register_loss("msn_wrapped", mode="embedding")
class MSNLossWrapper(nn.Module):
    def __init__(
        self,
        temp_anchor: float = 0.1,
        temp_target: float = 0.04,
        lambda_me_max: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.loss_fn = MSNLoss(
            temp_anchor=temp_anchor,
            temp_target=temp_target,
            lambda_me_max=lambda_me_max,
        )

    def forward(self, output: ModelOutput, target: Any = None) -> LossOutput:
        # Debug: print ModelOutput contents
        # print("DEBUG ModelOutput:", output)
        z_anchor = output.embeddings
        z_target = output.extra.get("z_target")
        prototypes = output.extra.get("prototypes")
        # print("DEBUG z_anchor:", type(z_anchor), getattr(z_anchor, 'shape', None))
        # print("DEBUG z_target:", type(z_target), getattr(z_target, 'shape', None))
        # print("DEBUG prototypes:", type(prototypes), getattr(prototypes, 'shape', None))
        if None in (z_anchor, z_target, prototypes):
            raise ValueError("Missing required fields in ModelOutput")
        # Cast to torch.Tensor for mypy
        from typing import cast

        z_anchor = cast(torch.Tensor, z_anchor)
        z_target = cast(torch.Tensor, z_target)
        prototypes = cast(torch.Tensor, prototypes)
        # Compute loss and components
        total_loss, components = self.loss_fn.compute_with_components(
            z_anchor, z_target, prototypes
        )

        return LossOutput(total=total_loss, components=dict(components))
