"""
Contains a set of dataset classes for different families of models.

Available dataset classes:
- ContrastiveDataset
- SuperResolutionDataset
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import requests

from refrakt_core.registry.dataset_registry import register_dataset


@register_dataset("contrastive")
class ContrastiveDataset(Dataset[Tuple[Tensor, Tensor, Any]]):
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
        train: Optional[bool] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

        if self.transform and hasattr(self.transform, "transforms"):
            self.transform.transforms = [
                t for t in self.transform.transforms if not isinstance(t, nn.Flatten)
            ]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Any]:
        item = self.base_dataset[idx]

        # Handle tuple-based dataset
        x = item[0] if isinstance(item, tuple) and len(item) >= 2 else item

        if self.transform:
            view1 = self.transform(x)
            view2 = self.transform(x)
            label = item[1] if isinstance(item, tuple) and len(item) >= 2 else -1
            # print(f"[DEBUG] Label: {label}")
            return view1, view2, label

        return x, x, -1


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
        transform: Optional[
            Callable[[Image.Image, Image.Image], Tuple[Tensor, Tensor]]
        ] = None,
        train: Optional[bool] = None,
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


@register_dataset("msn_contrastive")
class MSNCompatibleContrastiveDataset(Dataset):
    def __init__(
        self, base_dataset: Dataset, transform: Optional[Callable] = None, **kwargs
    ):
        self.dataset = ContrastiveDataset(
            base_dataset=base_dataset, transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, (tuple, list)):
            anchor, target = item[:2]
        else:
            anchor = target = item
        return {"anchor": anchor, "target": target}


@register_dataset("tabular_ml")
class TabularMLDataset:
    """
    Dataset for tabular ML tasks. Loads a CSV and returns X, y as numpy arrays.
    Args:
        csv_path (str): Path to the CSV file.
        target_col (str): Name of the target column.
        drop_cols (Optional[list[str]]): Columns to drop (besides target).
        download_url (Optional[str]): URL to download the CSV if not present.
    """
    def __init__(self, csv_path: str, target_col: str, drop_cols: Optional[list] = None, download_url: Optional[str] = None, **kwargs):
        if not os.path.exists(csv_path) and download_url:
            print(f"[TabularMLDataset] Downloading dataset from {download_url} to {csv_path}...")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            r = requests.get(download_url)
            r.raise_for_status()
            with open(csv_path, 'wb') as f:
                f.write(r.content)
            print(f"[TabularMLDataset] Download complete.")
        df = pd.read_csv(csv_path)
        if drop_cols:
            df = df.drop(columns=drop_cols)
        self.y = df[target_col].values
        self.X = df.drop(columns=[target_col]).values
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    def get_numpy(self):
        return self.X, self.y
