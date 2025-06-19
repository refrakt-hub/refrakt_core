import torch
from torch import nn
from typing import Any, Dict, Union, Optional

from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.losses.vae import VAELossWrapper
from refrakt_core.wrappers.losses.mae import MAELossWrapper

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
        **kwargs: Any
    ) -> LossOutput:
        if self.is_mae:
            return self.loss_fn(output)

        if self.is_vae:
            return self.loss_fn(output, target if target is not None else output.reconstruction)

        if isinstance(output, ModelOutput):
            if hasattr(output, "logits") and output.logits is not None:
                output_tensor = output.logits
            elif hasattr(output, "reconstruction") and output.reconstruction is not None:
                output_tensor = output.reconstruction
            elif hasattr(output, "features") and output.features is not None:
                output_tensor = output.features
            else:
                raise ValueError("Cannot extract tensor from ModelOutput")
        else:
            output_tensor = output

        effective_target = target if target is not None else output_tensor
        result = self.loss_fn(output_tensor, effective_target)

        if isinstance(result, LossOutput):
            return result
        elif isinstance(result, torch.Tensor):
            return LossOutput(total=result, components={"loss": result})
        else:
            raise TypeError(f"[DefaultLossWrapper] Unexpected loss_fn return type: {type(result)}")
