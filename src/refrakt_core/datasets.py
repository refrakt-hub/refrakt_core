"""
Contains a set of dataset classes for different families of models.

Available dataset classes:
- ContrastiveDataset
- SuperResolutionDataset
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sized, Tuple, Union, cast

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset

from refrakt_core.loaders.dataset_loader import load_custom_dataset
from refrakt_core.registry.dataset_registry import register_dataset


@register_dataset("contrastive")
class ContrastiveDataset(Dataset[Any]):
    """
    Dataset wrapper for contrastive learning methods like SimCLR and DINO.

    Args:
        base_dataset (Dataset): The underlying dataset to wrap.
        transform (Optional[Callable]): A torchvision-style transform callable.
        train (Optional[bool]): Flag indicating training mode (unused, for compatibility).
    """

    def __init__(
        self,
        base_dataset: Any,
        transform: Optional[Callable[[Any], Tensor]] = None,
        train: Optional[bool] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

        # Handle transform with transforms attribute (like torchvision.Compose)
        if self.transform and hasattr(self.transform, "transforms"):
            transforms_attr = getattr(self.transform, "transforms", None)
            if transforms_attr and hasattr(transforms_attr, "__iter__"):
                filtered = [t for t in transforms_attr if not isinstance(t, nn.Flatten)]
                self.transform.transforms = filtered

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore

    def __getitem__(self, idx: int) -> Any:
        item = self.base_dataset[idx]  # type: ignore

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
class MSNCompatibleContrastiveDataset(Dataset[Any]):
    def __init__(
        self,
        base_dataset: Any,
        transform: Optional[Callable[[Any], Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.dataset: Any = ContrastiveDataset(
            base_dataset=base_dataset, transform=transform
        )

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, idx: int) -> Any:
        item = self.dataset[idx]
        if isinstance(item, (tuple, list)):
            anchor, target = item[:2]
        else:
            anchor = target = item
        return {"anchor": anchor, "target": target}


@register_dataset("custom")
class CustomDataset(Dataset[Any]):
    """
    Custom dataset that loads data from zip files with automatic format detection.

    This dataset integrates the custom zip file loading functionality with the registry system.

    Args:
        zip_path (str): Path to the zip file containing the dataset
        task_type (Optional[str]): Type of task (gan, supervised, contrastive). If None, auto-detects.
        transform (Optional[Callable]): Transform to apply to images
        train (bool): Whether this is for training (True) or validation (False)
        **kwargs: Additional arguments passed to the underlying dataset loader
    """

    def __init__(
        self,
        zip_path: str,
        task_type: Optional[str] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        train: bool = True,
        **kwargs: Any,
    ) -> None:
        self.zip_path = zip_path
        self.task_type = task_type
        self.transform = transform
        self.train = train
        self.kwargs = kwargs

        # Load the custom dataset using the existing loader
        # Note: load_custom_dataset expects torchvision.Compose for transform
        # but we'll pass it as-is and let the loader handle it
        train_dataset, val_dataset = load_custom_dataset(
            zip_path=zip_path,
            task_type=task_type,
            transform=transform,  # type: ignore
            **kwargs,
        )

        # Select the appropriate dataset based on train flag
        if train:
            self.dataset = train_dataset
        else:
            self.dataset = val_dataset if val_dataset is not None else train_dataset

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[idx]


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

    def __init__(
        self,
        csv_path: str,
        target_col: str,
        drop_cols: Optional[List[str]] = None,
        download_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not os.path.exists(csv_path) and download_url:
            print(
                f"[TabularMLDataset] Downloading dataset from {download_url} to {csv_path}..."
            )
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            r = requests.get(download_url)
            r.raise_for_status()
            with open(csv_path, "wb") as f:
                f.write(r.content)
            print("[TabularMLDataset] Download complete.")
        df = pd.read_csv(csv_path)
        if drop_cols:
            df = df.drop(columns=drop_cols)
        self.y: Any = df[target_col].values  # type: ignore
        self.X: Any = df.drop(columns=[target_col]).values  # type: ignore

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        return self.X[idx], self.y[idx]

    def get_numpy(self) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        return self.X, self.y


@register_dataset("dummy")
class DummyDataset(Dataset[Tuple[Tensor, int]]):
    """
    Dummy dataset for testing purposes.

    Args:
        size (int): Number of samples in the dataset
        transform (Optional[Callable]): Transform to apply
        train (Optional[bool]): Whether this is for training (unused, for compatibility)
    """

    def __init__(
        self,
        size: int = 100,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        train: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.size: int = size
        self.transform: Optional[Callable[[Tensor], Tensor]] = transform
        self.data: Tensor = torch.randn(size, 3, 32, 32)  # Random images
        self.targets: Tensor = torch.randint(0, 10, (size,))  # Random labels

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        data: Tensor = self.data[idx]
        target: int = int(self.targets[idx].item())

        if self.transform:
            data = self.transform(data)

        return data, target
