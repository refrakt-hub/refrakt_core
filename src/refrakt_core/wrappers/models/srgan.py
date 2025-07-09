from typing import Any, Callable, Dict, Optional, Union, cast

import torch

from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("srgan")
class SRGANWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.model = model
        self.wrapper_config = kwargs

    def generate(self, input_data: torch.Tensor) -> ModelOutput:
        sr_image = self.model.generate(input_data)
        return ModelOutput(
            image=sr_image,
            extra={
                "wrapper_type": "srgan",
                "scale_factor": getattr(self.model, "scale_factor", None),
                **self.wrapper_config,
            },
        )

    def discriminate(self, input_data: torch.Tensor) -> ModelOutput:
        logits = self.model.discriminate(input_data)
        return ModelOutput(
            logits=logits, extra={"wrapper_type": "srgan", **self.wrapper_config}
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        return self.generate(x)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: Optional[Dict[str, torch.optim.Optimizer]] = None,
        loss_fn: Optional[Dict[str, Callable[..., Any]]] = None,
        device: str = "cuda",
        phase: Optional[str] = None,
    ) -> Dict[str, LossOutput]:
        lr = batch["lr"]
        hr = batch["hr"]
        if phase == "discriminator":
            with torch.no_grad():
                sr = self.model.generate(lr).detach()
            pred_real = ModelOutput(
                logits=self.model.discriminate(hr), extra={"target_is_real": True}
            )
            pred_fake = ModelOutput(
                logits=self.model.discriminate(sr), extra={"target_is_real": False}
            )
            assert loss_fn is not None and "discriminator" in loss_fn
            real_loss = loss_fn["discriminator"](pred_real)
            fake_loss = loss_fn["discriminator"](pred_fake)
            total = real_loss.total + fake_loss.total
            return {
                "d_loss": LossOutput(
                    total=total,
                    components={"real": real_loss.total, "fake": fake_loss.total},
                )
            }
        elif phase == "generator":
            sr = self.model.generator(lr)
            pred_fake = self.model.discriminator(sr)
            fake_output = ModelOutput(logits=pred_fake, extra={"target_is_real": True})
            assert loss_fn is not None and "discriminator" in loss_fn
            gan_loss = loss_fn["discriminator"](fake_output)
            pixel_loss: Optional[LossOutput] = None
            total = gan_loss.total
            if loss_fn is not None and "generator" in loss_fn:
                raw_pixel_loss = loss_fn["generator"](sr, hr)
                pixel_loss = (
                    raw_pixel_loss
                    if isinstance(raw_pixel_loss, LossOutput)
                    else LossOutput(
                        total=raw_pixel_loss, components={"pixel": raw_pixel_loss}
                    )
                )
                total += pixel_loss.total
            components = {"gan": gan_loss.total}
            if pixel_loss is not None:
                components["pixel"] = pixel_loss.total
            return {"g_loss": LossOutput(total=total, components=components)}
        else:
            raise ValueError(
                "❌ 'phase' must be either 'generator' or 'discriminator' in SRGANWrapper.training_step()"
            )

    @property
    def generator(self) -> Any:
        return self.model.generator

    @property
    def discriminator(self) -> Any:
        return self.model.discriminator

    def forward_for_graph(
        self, x: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Handles input for graph logging — extracts 'lr' from dict or creates dummy input.
        """
        if isinstance(x, dict):
            x = x.get("lr")
            if x is None:
                raise ValueError(
                    "forward_for_graph(): 'lr' key not found in input dict"
                )
        if x is None:
            x = torch.randn(1, 3, 24, 24).to(next(self.model.parameters()).device)
        output = self.forward(x)
        return cast(torch.Tensor, output.image)
