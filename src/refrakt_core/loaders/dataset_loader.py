"""
Dynamic dataset loader for Refrakt with support for custom zip files and torchvision datasets.

This module provides a flexible system for loading datasets from custom zip files
and torchvision datasets with automatic format detection and validation.
"""

import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms  # type: ignore

from refrakt_core.loaders.utils import find_directory_by_keywords
from refrakt_core.logging_config import get_logger


@dataclass
class DatasetFormat:
    """Represents a dataset format specification."""

    name: str
    description: str
    structure: Dict[str, Any]
    supported_tasks: List[str]
    validation_rules: List[Callable[[Path], bool]]


def validate_gan_structure(extracted_path: Path) -> bool:
    """Validate GAN dataset structure."""
    lr_dir = extracted_path / "lr"
    hr_dir = extracted_path / "hr"

    if not lr_dir.exists() or not hr_dir.exists():
        return False

    lr_files = (
        list(lr_dir.glob("*.png"))
        + list(lr_dir.glob("*.jpg"))
        + list(lr_dir.glob("*.jpeg"))
    )
    hr_files = (
        list(hr_dir.glob("*.png"))
        + list(hr_dir.glob("*.jpg"))
        + list(hr_dir.glob("*.jpeg"))
    )

    if not lr_files or not hr_files:
        return False

    # Check if file names match (allowing for different extensions)
    lr_names = {f.stem for f in lr_files}
    hr_names = {f.stem for f in hr_files}

    return bool(lr_names & hr_names)  # Check for intersection


def _find_train_directory(extracted_path: Path) -> Optional[Path]:
    """Find a directory that contains 'train' in its name."""
    logger = get_logger("dataset_loader")
    logger.debug(f"Searching for train directory in: {extracted_path}")
    keywords = ["train", "traing", "trainging", "training"]
    result = find_directory_by_keywords(extracted_path, keywords)
    if result:
        logger.debug(f"Found train directory: {result}")
    else:
        logger.debug("No train directory found")
    return result


def _find_val_directory(extracted_path: Path) -> Optional[Path]:
    """Find a directory that contains 'val' or 'test' in its name."""
    logger = get_logger("dataset_loader")
    logger.debug(f"Searching for val/test directory in: {extracted_path}")
    keywords = ["val", "test", "testing", "validation"]
    result = find_directory_by_keywords(extracted_path, keywords)
    if result:
        logger.debug(f"Found val/test directory: {result}")
    else:
        logger.debug("No val/test directory found")
    return result


def validate_supervised_structure(extracted_path: Path) -> bool:
    """Validate supervised dataset structure."""
    # Try to find train directory with flexible naming
    train_dir = _find_train_directory(extracted_path)

    if train_dir is None:
        # Fallback to exact "train" directory
        train_dir = extracted_path / "train"
        if not train_dir.exists():
            return False

    # Check for class directories in train
    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    if not train_classes:
        return False

    # Check if each class directory has images
    for class_dir in train_classes:
        images = (
            list(class_dir.glob("*.png"))
            + list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
        )
        if not images:
            return False

    return True


def validate_contrastive_structure(extracted_path: Path) -> bool:
    """Validate contrastive dataset structure."""
    images_dir = extracted_path / "images"

    if not images_dir.exists():
        return False

    images = (
        list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
    )
    return len(images) > 0


def detect_dataset_format(extracted_path: Path) -> Optional[str]:
    """
    Detect the format of a dataset.

    Args:
        extracted_path: Path to the extracted dataset

    Returns:
        Format name if detected, None otherwise
    """
    logger = get_logger("dataset_detector")

    # Check GAN format
    if validate_gan_structure(extracted_path):
        logger.info("Detected format: gan")
        return "gan"

    # Check supervised format
    if validate_supervised_structure(extracted_path):
        logger.info("Detected format: supervised")
        return "supervised"

    # Check contrastive format
    if validate_contrastive_structure(extracted_path):
        logger.info("Detected format: contrastive")
        return "contrastive"

    return None


def validate_image_size(
    image_path: Path, max_size: Tuple[int, int] = (224, 224), max_ratio: float = 2.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate an image for size constraints.

    Args:
        image_path: Path to the image file
        max_size: Maximum allowed image size (width, height)
        max_ratio: Maximum ratio of image size to max_size before throwing error

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # Check if image is too large
            if width > max_size[0] * max_ratio or height > max_size[1] * max_ratio:
                error_msg = (
                    f"Image {image_path} is too large ({width}x{height}). "
                    f"Maximum allowed size is {max_size[0] * max_ratio}x{max_size[1] * max_ratio}. "
                    f"Please resize the image to {max_size[0]}x{max_size[1]} or smaller."
                )
                return False, error_msg

            return True, None

    except Exception as e:
        return False, f"Failed to validate image {image_path}: {e}"


def extract_zip_file(zip_path: Path) -> Path:
    """Extract a zip file to a temporary directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="refrakt_dataset_"))

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find the actual dataset directory (handle nested structures)
    extracted_path = _find_dataset_directory(temp_dir)
    logger = get_logger("dataset_loader")
    logger.info(f"Extracted dataset to: {extracted_path}")

    return extracted_path


def _find_dataset_directory(temp_dir: Path) -> Path:
    """Find the actual dataset directory within the extracted structure."""
    if (temp_dir / "lr").exists() and (temp_dir / "hr").exists():
        return temp_dir
    if (temp_dir / "train").exists():
        return temp_dir
    dataset_names = ["data", "dataset", "images", "train", "lr"]
    for name in dataset_names:
        potential_path = temp_dir / name
        if potential_path.exists():
            return potential_path
    for item in temp_dir.iterdir():
        if item.is_dir():
            if _has_training_data(item):
                return item
    return temp_dir


def _has_training_data(directory: Path) -> bool:
    """Check if a directory contains training data structure."""
    subdirs: List[Path] = [d for d in directory.iterdir() if d.is_dir()]

    # If we have subdirectories, check if they contain images
    if subdirs:
        for subdir in subdirs[:3]:  # Check first 3 subdirs
            images: List[Path] = (
                list(subdir.glob("*.png"))
                + list(subdir.glob("*.jpg"))
                + list(subdir.glob("*.jpeg"))
            )
            if images:
                return True

    return False


def validate_dataset_images(
    dataset_path: Path, max_size: Tuple[int, int] = (224, 224)
) -> None:
    """Validate all images in the dataset."""
    image_files: List[Path] = []

    # Find all image files
    for ext in [".png", ".jpg", ".jpeg"]:
        image_files.extend(dataset_path.rglob(f"*{ext}"))

    if not image_files:
        raise ValueError("No image files found in dataset")

    # Validate each image
    invalid_images: List[Tuple[Path, Optional[str]]] = []
    for image_path in image_files:
        is_valid, error_msg = validate_image_size(image_path, max_size)
        if not is_valid:
            invalid_images.append((image_path, error_msg))

    if invalid_images:
        error_messages = "\n".join(
            [f"{path}: {msg}" for path, msg in invalid_images[:5]]
        )
        if len(invalid_images) > 5:
            error_messages += f"\n... and {len(invalid_images) - 5} more images"

        raise ValueError(f"Found invalid images:\n{error_messages}")


def load_custom_dataset(
    zip_path: Union[str, Path],
    task_type: Optional[str] = None,
    transform: Optional[transforms.Compose] = None,
    **kwargs: Any,
) -> Tuple[Dataset[Any], Optional[Dataset[Any]]]:
    """
    Load a custom dataset from a zip file.

    Args:
        zip_path: Path to the zip file
        task_type: Type of task (gan, supervised, contrastive)
        transform: Optional transform to apply
        **kwargs: Additional arguments for dataset creation

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    # Extract zip file
    extracted_path: Path = extract_zip_file(zip_path)

    # Detect format if not specified
    if task_type is None:
        task_type = detect_dataset_format(extracted_path)
        if task_type is None:
            raise ValueError(
                "Could not detect dataset format. Available formats: gan, supervised, contrastive"
            )

    # Validate images
    validate_dataset_images(extracted_path)

    # Create dataset based on format
    if task_type == "gan":
        return _create_gan_dataset(extracted_path, transform, **kwargs)
    elif task_type == "supervised":
        return _create_supervised_dataset(extracted_path, transform, **kwargs)
    elif task_type == "contrastive":
        return _create_contrastive_dataset(extracted_path, transform, **kwargs)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def _create_gan_dataset(
    dataset_path: Path, transform: Optional[transforms.Compose], **kwargs: Any
) -> Tuple[Dataset[Any], Optional[Dataset[Any]]]:
    """Create a GAN dataset from the extracted files."""
    from refrakt_core.datasets import SuperResolutionDataset

    lr_dir: Path = dataset_path / "lr"
    hr_dir: Path = dataset_path / "hr"

    if not lr_dir.exists() or not hr_dir.exists():
        raise ValueError("GAN dataset must have 'lr' and 'hr' directories")

    # Create train dataset
    train_dataset: Dataset[Any] = SuperResolutionDataset(
        lr_dir=lr_dir, hr_dir=hr_dir, transform=transform, train=True
    )

    # Create validation dataset if val directory exists
    val_dataset: Optional[Dataset[Any]] = None
    val_lr_dir: Path = dataset_path / "val" / "lr"
    val_hr_dir: Path = dataset_path / "val" / "hr"

    if val_lr_dir.exists() and val_hr_dir.exists():
        val_dataset = SuperResolutionDataset(
            lr_dir=val_lr_dir, hr_dir=val_hr_dir, transform=transform, train=False
        )

    return train_dataset, val_dataset


def _create_supervised_dataset(
    dataset_path: Path, transform: Optional[transforms.Compose], **kwargs: Any
) -> Tuple[Dataset[Any], Optional[Dataset[Any]]]:
    """Create a supervised dataset from the extracted files."""
    train_dir: Optional[Path] = _find_train_directory(dataset_path)
    print(f"[DEBUG] train_dir: {train_dir}")
    if train_dir is None:
        train_dir = dataset_path / "train"
    if not train_dir.exists():
        raise ValueError(
            "Supervised dataset must have a directory containing 'train' in its name"
        )
    val_dir: Optional[Path] = _find_val_directory(dataset_path)
    if val_dir is None:
        val_dir = dataset_path / "val"
    train_dataset: Dataset[Any] = datasets.ImageFolder(
        root=train_dir, transform=transform
    )
    val_dataset: Optional[Dataset[Any]] = None
    if val_dir.exists():
        val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    return train_dataset, val_dataset


def _create_contrastive_dataset(
    dataset_path: Path, transform: Optional[transforms.Compose], **kwargs: Any
) -> Tuple[Dataset[Any], Optional[Dataset[Any]]]:
    """Create a contrastive dataset from the extracted files."""
    from refrakt_core.datasets import ContrastiveDataset

    images_dir: Path = dataset_path / "images"

    if not images_dir.exists():
        raise ValueError("Contrastive dataset must have an 'images' directory")

    # Create a simple dataset that returns image paths
    class ImagePathDataset(Dataset[Any]):
        def __init__(self, images_dir: Path) -> None:
            self.images_dir: Path = images_dir
            self.image_files: List[Path] = (
                list(images_dir.glob("*.png"))
                + list(images_dir.glob("*.jpg"))
                + list(images_dir.glob("*.jpeg"))
            )

        def __len__(self) -> int:
            return len(self.image_files)

        def __getitem__(self, idx: int) -> Path:
            return self.image_files[idx]

    base_dataset: Dataset[Any] = ImagePathDataset(images_dir)

    # Create train dataset
    train_dataset: Dataset[Any] = ContrastiveDataset(
        base_dataset=base_dataset, transform=transform, train=True
    )

    # Create validation dataset if val directory exists
    val_dataset: Optional[Dataset[Any]] = None
    val_dir: Path = dataset_path / "val"

    if val_dir.exists():
        val_base_dataset: Dataset[Any] = ImagePathDataset(val_dir)
        val_dataset = ContrastiveDataset(
            base_dataset=val_base_dataset, transform=transform, train=False
        )

    return train_dataset, val_dataset


def load_torchvision_dataset(
    dataset_name: str,
    root: str = "./data",
    train: bool = True,
    transform: Optional[transforms.Compose] = None,
    download: bool = True,
    **kwargs: Any,
) -> Dataset[Any]:
    """
    Load a torchvision dataset.

    Args:
        dataset_name: Name of the dataset (mnist, cifar10, cifar100, stl10, imagenet)
        root: Root directory for dataset
        train: Whether to load training set
        transform: Optional transform to apply
        download: Whether to download the dataset
        **kwargs: Additional arguments for dataset

    Returns:
        Dataset instance
    """
    dataset_map = {
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "stl10": datasets.STL10,
        "imagenet": datasets.ImageNet,
    }

    if dataset_name not in dataset_map:
        available = list(dataset_map.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Available: {available}"
        )

    dataset_cls = dataset_map[dataset_name]

    # Special handling for STL10 (no train parameter)
    if dataset_name == "stl10":
        split = "train" if train else "test"
        return cast(
            Dataset[Any],
            dataset_cls(
                root=root, split=split, transform=transform, download=download, **kwargs
            ),
        )

    # Special handling for ImageNet (requires specific structure)
    elif dataset_name == "imagenet":
        if not os.path.exists(os.path.join(root, "train")):
            raise ValueError(
                "ImageNet dataset not found. Please ensure the dataset is properly structured."
            )
        return cast(
            Dataset[Any],
            dataset_cls(
                root=root,
                split="train" if train else "val",
                transform=transform,
                **kwargs,
            ),
        )

    # Standard datasets
    else:
        return cast(
            Dataset[Any],
            dataset_cls(
                root=root, train=train, transform=transform, download=download, **kwargs
            ),
        )


def create_dataloader(
    dataset: Dataset[Any],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs: Any,
) -> DataLoader[Any]:
    """
    Create a DataLoader from a dataset.

    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )


def load_dataset(
    dataset_path: Union[str, Path],
    dataset_type: str = "auto",
    transform: Optional[transforms.Compose] = None,
    **kwargs: Any,
) -> Tuple[Dataset[Any], Optional[Dataset[Any]]]:
    """
    Load a dataset (custom or torchvision).

    Args:
        dataset_path: Path to dataset (zip file for custom, name for torchvision)
        dataset_type: Type of dataset (auto, custom, torchvision)
        transform: Optional transform to apply
        **kwargs: Additional arguments

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    dataset_path = str(dataset_path)

    # Auto-detect dataset type
    if dataset_type == "auto":
        if dataset_path.endswith(".zip"):
            dataset_type = "custom"
        elif dataset_path.lower() in [
            "mnist",
            "cifar10",
            "cifar100",
            "stl10",
            "imagenet",
        ]:
            dataset_type = "torchvision"
        else:
            dataset_type = "custom"

    if dataset_type == "custom":
        return load_custom_dataset(dataset_path, transform=transform, **kwargs)
    elif dataset_type == "torchvision":
        train_dataset: Dataset[Any] = load_torchvision_dataset(
            dataset_path, train=True, transform=transform, **kwargs
        )
        val_dataset: Optional[Dataset[Any]] = load_torchvision_dataset(
            dataset_path, train=False, transform=transform, **kwargs
        )
        return train_dataset, val_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
