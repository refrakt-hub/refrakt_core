"""
Contains a set of transform classes for specific use-cases.
Available transforms are:
- PairedTransform
- FlattenTransform
- PatchifyTransform
- PairedResize
"""

import random
from typing import Any, Optional, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch import Tensor

from refrakt_core.registry.transform_registry import register_transform


@register_transform("paired")
class PairedTransform:
    """
    A transform class for SR-based training that handles paired LR/HR images.
    Can optionally resize images before applying other transforms.
    """

    def __init__(
        self,
        crop_size: int = 96,
        resize_lr: Optional[Tuple[int, int]] = None,
        resize_hr: Optional[Tuple[int, int]] = None,
        scale_factor: int = 4,
    ) -> None:
        self.crop_size = crop_size
        self.scale_factor = scale_factor

        # Set up resize transforms if specified
        self.lr_resize = None
        self.hr_resize = None

        if resize_lr is not None:
            self.lr_resize = T.Resize(
                resize_lr, interpolation=T.InterpolationMode.BICUBIC
            )

        if resize_hr is not None:
            self.hr_resize = T.Resize(
                resize_hr, interpolation=T.InterpolationMode.BICUBIC
            )
        elif resize_lr is not None:
            # If only LR resize is specified, calculate HR size using scale factor
            hr_size = (resize_lr[0] * scale_factor, resize_lr[1] * scale_factor)
            self.hr_resize = T.Resize(
                hr_size, interpolation=T.InterpolationMode.BICUBIC
            )

    def __call__(self, lr: Any, hr: Any) -> Tuple[Tensor, Tensor]:
        # Apply resize if specified
        if self.lr_resize is not None:
            lr = self.lr_resize(lr)
        if self.hr_resize is not None:
            hr = self.hr_resize(hr)

        # Apply cropping
        i, j, h, w = T.RandomCrop.get_params(
            hr, output_size=(self.crop_size * 4, self.crop_size * 4)
        )
        hr = F.crop(hr, i, j, h, w)
        lr = F.crop(lr, i // 4, j // 4, h // 4, w // 4)

        # Apply random horizontal flip
        if random.random() > 0.5:
            hr = F.hflip(hr)
            lr = F.hflip(lr)

        # Convert to tensors
        hr = T.ToTensor()(hr)
        lr = T.ToTensor()(lr)

        return lr, hr


@register_transform("flatten")
class FlattenTransform:
    """
    A wrapper class that wraps around torch.flatten for a given tensor.
    """

    def __call__(self, x: Any) -> Any:
        return torch.flatten(x)


@register_transform("patchify")
class PatchifyTransform:
    """
    A transform class for ViT / Swin-based training.
    """

    def __init__(self, patch_size: int) -> None:
        self.patch_size = patch_size

    def __call__(self, img: Tensor) -> Tensor:
        _, h, w = img.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0, "Image dims must be divisible by patch size"
        return img


# ONLY FOR TESTING PURPOSES
@register_transform("dummy")
class DummyTransform:
    """
    Dummy transform for testing purposes.

    Args:
        **kwargs: Any additional arguments (ignored)
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __call__(self, x: Any) -> Any:
        return x


@register_transform("PairedTransform")
class PairedTransformWrapper:
    """
    Wrapper for PairedTransform to match the expected name in tests.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.transform = PairedTransform(**kwargs)

    def __call__(self, lr: Any, hr: Any) -> Any:
        return self.transform(lr, hr)
