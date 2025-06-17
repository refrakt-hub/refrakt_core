# refrakt_core/wrappers/srgan_wrapper.py

import torch
from torch import Tensor, nn
from typing import Dict, Any, Optional

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.registry.model_registry import get_model


@register_wrapper("srgan_wrapper")
class SRGANWrapper(nn.Module):
    """
    Wrapper for SRGAN that returns ModelOutput for standardized explainability + visualization.

    Outputs:
        - image: SR output from generator
        - extra:
            - low_res: original input (for reconstruction comparison)
            - discriminator_output: raw logits from D
            - realism_score: sigmoid(D output)
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[Dict[str, Any]] = None,
        modules: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.model = get_model(model_name, **(model_params or {}))

    def forward(self, x: Tensor, **kwargs) -> ModelOutput:
        # Generator output
        sr = self.model.generate(x)

        # Discriminator output for the generated SR image
        d_out = self.model.discriminate(sr)
        realism_score = torch.sigmoid(d_out)

        return ModelOutput(
            image=sr,
            extra={
                "low_res": x,  # useful for grid comparisons: lr → sr → hr
                "discriminator_output": d_out,
                "realism_score": realism_score,
            }
        )
