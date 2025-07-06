from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.losses.mae import MAELossWrapper
from refrakt_core.wrappers.losses.vae import VAELossWrapper
from refrakt_core.wrappers.utils.default_loss_utils import (
    handle_mae_loss,
    handle_vae_loss,
    extract_tensor_from_model_output,
    create_loss_output,
)


class DefaultLossWrapper(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.is_mae = isinstance(loss_fn, MAELossWrapper)
        self.is_vae = isinstance(loss_fn, VAELossWrapper)

    def forward(
        self,
        output: Union[torch.Tensor, ModelOutput, Dict],
        target: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> LossOutput:
        if self.is_mae:
            return handle_mae_loss(self.loss_fn, output)

        if self.is_vae:
            return handle_vae_loss(self.loss_fn, output, target)

        if isinstance(output, ModelOutput):
            output_tensor = extract_tensor_from_model_output(output)
        else:
            output_tensor = output

        effective_target = target if target is not None else output_tensor
        result = self.loss_fn(output_tensor, effective_target)

        return create_loss_output(result)
