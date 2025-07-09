"""
SRGAN model implementation for super-resolution using a GAN framework.

This module defines a Generator-Discriminator architecture trained using
adversarial and perceptual losses to upscale low-resolution images.
"""

from typing import Any, Dict

import torch
from torch import Tensor

from refrakt_core.models.templates.models import BaseGAN
from refrakt_core.registry.model_registry import register_model
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.utils.classes.srgan import Discriminator, Generator


@register_model("srgan")
class SRGAN(BaseGAN):
    """
    Super-Resolution Generative Adversarial Network (SRGAN).

    This model combines a generator and discriminator to perform
    super-resolution tasks on images.
    """

    def __init__(self, scale_factor: int = 4, model_name: str = "srgan") -> None:
        """
        Initialize the SRGAN model.

        Args:
            scale_factor: The upscaling factor for super-resolution. Defaults to 4.
            model_name: Model name. Defaults to "srgan".
        """
        super().__init__(model_name=model_name)
        self.scale_factor: int = scale_factor
        self.generator: Any = Generator(scale_factor=scale_factor)
        self.discriminator: Any = Discriminator()

    # Removed device property to avoid property/attribute conflict with BaseModel

    def training_step(
        self,
        batch: Dict[str, Tensor],
        optimizer: Dict[str, torch.optim.Optimizer],
        loss_fn: Dict[str, torch.nn.Module],
        device: torch.device,
    ) -> Dict[str, float]:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        # Generator update
        optimizer["generator"].zero_grad()
        sr = self.generator(lr)
        g_loss = loss_fn["generator"](sr, hr)
        g_loss.backward()
        optimizer["generator"].step()

        # Discriminator update
        optimizer["discriminator"].zero_grad()
        real_pred = self.discriminator(hr)
        fake_pred = self.discriminator(sr.detach())

        # FIX: Create ModelOutput with logits and target_is_real flag
        real_output = ModelOutput(logits=real_pred, extra={"target_is_real": True})
        fake_output = ModelOutput(logits=fake_pred, extra={"target_is_real": False})

        loss_real = loss_fn["discriminator"](real_output)
        loss_fake = loss_fn["discriminator"](fake_output)
        d_loss = 0.5 * (loss_real + loss_fake)

        d_loss.backward()
        optimizer["discriminator"].step()

        return {"g_loss": g_loss.item(), "d_loss": d_loss.item()}

    def generate(self, input_data: Tensor) -> Tensor:
        """
        Generate a super-resolution image from a low-resolution input.

        Args:
            input_data: Low-resolution input image.

        Returns:
            Super-resolution output image.
        """
        self.generator.eval()
        with torch.no_grad():
            if input_data.device != self.device:
                input_data = input_data.to(self.device)
            return self.generator(input_data)  # type: ignore[no-any-return]

    def discriminate(self, input_data: Tensor) -> Tensor:
        """
        Discriminate between real and fake images.

        Args:
            input_data: Input image.

        Returns:
            Probability that the input is a real image.
        """
        # self.discriminator.eval()
        # with torch.no_grad():
        #     if input_data.device != self.device:
        #         input_data = input_data.to(self.device)
        #     return self.discriminator(input_data)
        if input_data.device != self.device:
            input_data = input_data.to(self.device)
        return self.discriminator(input_data)  # type: ignore[no-any-return]

    def summary(self) -> Dict[str, object]:
        """
        Get a summary of the SRGAN model including additional SR-specific information.

        Returns:
            Model summary information.
        """
        base_summary = super().summary()
        base_summary.update({"scale_factor": self.scale_factor})
        return base_summary

    def save_model(self, path: str) -> None:
        """
        Save model weights to disk with SR-specific attributes.

        Args:
            path: Path to save the model.
        """
        model_state = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "scale_factor": self.scale_factor,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
        }
        torch.save(model_state, path)
        print(f"SRGAN model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model weights from disk including SR-specific attributes.

        Args:
            path: Path to load the model from.
        """
        super().load_model(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.scale_factor = checkpoint.get("scale_factor", self.scale_factor)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the generator.

        Args:
            x: Input low-resolution image tensor.

        Returns:
            Super-resolution output image tensor.
        """
        return self.generator(x)  # type: ignore[no-any-return]
