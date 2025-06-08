"""
Perceptual loss implementation using a pre-trained VGG19 network.
"""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import vgg19

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

        vgg: nn.Sequential = vgg19(pretrained=True).features[:36].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg: nn.Sequential = vgg
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
        sr_features: Tensor = self.vgg(sr)
        hr_features: Tensor = self.vgg(hr)

        if sr_features.shape != hr_features.shape:
            raise ValueError(
                f"Feature shape mismatch: {sr_features.shape} vs {hr_features.shape}"
            )

        return F.mse_loss(sr_features, hr_features)

    def get_config(self) -> Dict[str, str]:
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
