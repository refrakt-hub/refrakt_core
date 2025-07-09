from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.losses.mae import MAELossWrapper
from refrakt_core.wrappers.losses.vae import VAELossWrapper
from refrakt_core.wrappers.utils.default_loss_utils import (
    create_loss_output,
    extract_tensor_from_model_output,
    handle_mae_loss,
    handle_vae_loss,
)


class DefaultLossWrapper(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.is_mae = isinstance(loss_fn, MAELossWrapper)
        self.is_vae = isinstance(loss_fn, VAELossWrapper)

    def forward(
        self,
        output: Union[torch.Tensor, ModelOutput, Dict[str, Any]],
        target: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> LossOutput:
        if self.is_mae:
            return handle_mae_loss(self.loss_fn, output)

        if self.is_vae:
            return handle_vae_loss(self.loss_fn, output, target)

        if isinstance(output, ModelOutput):
            output_tensor = extract_tensor_from_model_output(output)
        elif isinstance(output, torch.Tensor):
            output_tensor = output
        elif isinstance(output, dict):
            # If output is a dict, try to extract 'logits' or use as is
            output_tensor = output.get("logits", output)
        else:
            raise TypeError(f"Unsupported output type: {type(output)}")

        effective_target = target if target is not None else output_tensor
        result = self.loss_fn(output_tensor, effective_target)

        return create_loss_output(result)
