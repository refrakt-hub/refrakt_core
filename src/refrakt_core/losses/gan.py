"""
The GAN Loss implementation for adversarial training.
"""

from typing import Union

import torch
from torch import Tensor, nn

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss


@register_loss("gan")
class GANLoss(BaseLoss):
    """
    GAN Loss for adversarial training.

    Supports both:
    - Binary Cross Entropy (BCE) with logits (default)
    - Least Squares GAN (LSGAN)

    Args:
        use_lsgan (bool): Whether to use LSGAN (MSELoss) instead of BCEWithLogitsLoss.
        device (str): Device identifier string (e.g., "cuda" or "cpu").
    """

    def __init__(self, use_lsgan: bool = False, device: str = "cuda") -> None:
        super().__init__(name="GANLoss")
        self.loss: nn.Module = nn.MSELoss() if use_lsgan else nn.BCEWithLogitsLoss()
        self.use_lsgan: bool = use_lsgan
        self.device: torch.device = torch.device(device)

    def forward(self, pred: Tensor, target_is_real: bool) -> Tensor:
        """
        Compute the GAN loss for discriminator or generator output.

        Args:
            pred (Tensor): The prediction logits from the discriminator, shape (N, *).
            target_is_real (bool): Whether the target is real (True) or fake (False).

        Returns:
            Tensor: Scalar GAN loss value.

        Raises:
            TypeError: If input types are incorrect.
        """
        if not isinstance(pred, Tensor):
            raise TypeError("pred must be a torch.Tensor.")
        if not isinstance(target_is_real, bool):
            raise TypeError("target_is_real must be a boolean.")

        target: Tensor = (
            torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        )
        target = target.to(pred.device)
        return self.loss(pred, target)  # type: ignore[no-any-return]

    def get_config(self) -> dict[str, Union[str, bool]]:
        """
        Return configuration details of the GANLoss.

        Returns:
            dict: Configuration dictionary with loss type and device.
        """
        return {
            **super().get_config(),
            "use_lsgan": self.use_lsgan,
            "device": str(self.device),
        }
