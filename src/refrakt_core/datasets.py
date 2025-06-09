"""
Contains a set of dataset classes for different families of models.

Available dataset classes:
- ContrastiveDataset
- SuperResolutionDataset
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Dict, Any

from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset

from refrakt_core.registry.dataset_registry import register_dataset


@register_dataset("contrastive")
class ContrastiveDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Dataset wrapper for contrastive learning methods like SimCLR and DINO.

    Args:
        base_dataset (Dataset): The underlying dataset to wrap.
        transform (Optional[Callable]): A torchvision-style transform callable.
        train (Optional[bool]): Flag indicating training mode (unused, for compatibility).
    """
    def __init__(
        self,
        base_dataset: Dataset,
        transform: Optional[Callable[[Any], Tensor]] = None,
        train: Optional[bool] = None
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

        if self.transform and hasattr(self.transform, "transforms"):
            self.transform.transforms = [
                t for t in self.transform.transforms if not isinstance(t, nn.Flatten)
            ]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        item = self.base_dataset[idx]

        # Handle tuple-based dataset
        x = item[0] if isinstance(item, tuple) and len(item) >= 2 else item

        if self.transform:
            view1 = self.transform(x)
            view2 = self.transform(x)
            return view1, view2

        return x, x


@register_dataset("super_resolution")
class SuperResolutionDataset(Dataset[Dict[str, Tensor]]):
    """
    Dataset for super-resolution tasks. Loads paired LR and HR images.

    Args:
        lr_dir (Union[str, Path]): Path to low-resolution image directory.
        hr_dir (Union[str, Path]): Path to high-resolution image directory.
        transform (Optional[Callable]): Callable to apply joint transforms to (lr, hr) pair.
        train (Optional[bool]): Flag indicating training mode (unused, for compatibility).
    """
    def __init__(
        self,
        lr_dir: Union[str, Path],
        hr_dir: Union[str, Path],
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[Tensor, Tensor]]] = None,
        train: Optional[bool] = None
    ) -> None:
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.filenames = sorted(os.listdir(self.lr_dir))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        fname = self.filenames[idx]
        lr_img = Image.open(self.lr_dir / fname).convert("RGB")
        hr_img = Image.open(self.hr_dir / fname).convert("RGB")

        if self.transform:
            lr_tensor, hr_tensor = self.transform(lr_img, hr_img)
        else:
            raise ValueError("Transform must be provided for SuperResolutionDataset.")

        return {"lr": lr_tensor, "hr": hr_tensor}
