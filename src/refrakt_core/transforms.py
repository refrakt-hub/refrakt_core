"""
Contains a set of transform classes for specific use-cases.
Available transforms are:
- PairedTransform
- FlattenTransform
PatchifyTransform
"""

import random
from typing import Any, Tuple

import torch
import torchvision.transforms as T
from torch import Tensor

from refrakt_core.registry.transform_registry import register_transform


@register_transform("paired")
class PairedTransform:
    """
    A transform class for SR-based training.
    """

    def __init__(self, crop_size: int = 96) -> None:
        self.crop_size = crop_size

    def __call__(self, lr: Any, hr: Any) -> Tuple[Tensor, Tensor]:
        i, j, h, w = T.RandomCrop.get_params(
            hr, output_size=(self.crop_size * 4, self.crop_size * 4)
        )
        hr = T.functional.crop(hr, i, j, h, w)
        lr = T.functional.crop(lr, i // 4, j // 4, h // 4, w // 4)

        if random.random() > 0.5:
            hr = T.functional.hflip(hr)
            lr = T.functional.hflip(lr)

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
