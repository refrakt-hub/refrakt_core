"""
Perceptual loss implementation using a pre-trained VGG19 network.
"""

from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import vgg19  # type: ignore

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss


@register_loss("perceptual")
class PerceptualLoss(BaseLoss):
    """
    Perceptual Loss using a pre-trained VGG19 network.

    Computes the MSE loss between feature maps extracted from
    a fixed VGG19 model for the super-resolved and high-resolution images.
    """

    def __init__(self, device: str = "cuda") -> None:
        """
        Args:
            device (str): Device to load the VGG network onto.
        """
        super().__init__(name="PerceptualLoss")

        vgg_full: nn.Module = vgg19(pretrained=True).features
        # Extract features up to layer 36 (before the last maxpool)
        self.vgg = nn.Sequential(*list(vgg_full.children())[:36]).to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.device: str = device

        self.freeze()

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        """
        Compute perceptual loss between super-resolved and high-resolution images.

        Args:
            sr (Tensor): Super-resolved image of shape (N, C, H, W).
            hr (Tensor): High-resolution target image of shape (N, C, H, W).

        Returns:
            Tensor: Scalar loss computed as MSE between VGG19 feature maps.
        """
        # Move inputs to the same device as the VGG model
        sr = sr.to(self.device)
        hr = hr.to(self.device)

        sr_features: Tensor = self.vgg(sr)
        hr_features: Tensor = self.vgg(hr)

        if sr_features.shape != hr_features.shape:
            raise ValueError(
                f"Feature shape mismatch: {sr_features.shape} vs {hr_features.shape}"
            )

        return F.mse_loss(sr_features, hr_features)

    def get_config(self) -> dict[str, Any]:
        """
        Get the configuration of the loss function.

        Returns:
            dict: Configuration dictionary with model and layer info.
        """
        return {
            **super().get_config(),
            "backbone": "vgg19",
            "layers_used": "features[:36]",
            "device": self.device,
        }
