"""
Dynamic dataset loader for Refrakt with support for custom zip files.

This module provides a flexible system for loading datasets from custom zip files
with support for different formats (GAN, supervised) and automatic format detection.
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore

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


class DatasetFormatDetector:
    """Detects the format of a dataset from its structure."""

    def __init__(self) -> None:
        self.logger = get_logger("dataset_detector")
        self.formats = self._register_formats()

    def _register_formats(self) -> Dict[str, DatasetFormat]:
        """Register known dataset formats."""
        formats = {}

        # GAN format (like DIV2K)
        gan_format = DatasetFormat(
            name="gan",
            description="GAN training format with paired LR/HR images",
            structure={
                "required_dirs": ["lr", "hr"],
                "optional_dirs": ["val", "test"],
                "file_extensions": [".png", ".jpg", ".jpeg"],
                "naming_convention": "matching_names",
            },
            supported_tasks=["super_resolution", "gan"],
            validation_rules=[self._validate_gan_structure],
        )
        formats["gan"] = gan_format

        # Supervised classification format
        supervised_format = DatasetFormat(
            name="supervised",
            description="Supervised classification format with class directories",
            structure={
                "required_dirs": ["train", "val"],
                "optional_dirs": ["test"],
                "file_extensions": [".png", ".jpg", ".jpeg"],
                "naming_convention": "class_directories",
            },
            supported_tasks=["classification", "supervised"],
            validation_rules=[self._validate_supervised_structure],
        )
        formats["supervised"] = supervised_format

        # Contrastive learning format
        contrastive_format = DatasetFormat(
            name="contrastive",
            description="Contrastive learning format with single directory",
            structure={
                "required_dirs": ["images"],
                "optional_dirs": ["val", "test"],
                "file_extensions": [".png", ".jpg", ".jpeg"],
                "naming_convention": "flat_structure",
            },
            supported_tasks=["contrastive", "self_supervised"],
            validation_rules=[self._validate_contrastive_structure],
        )
        formats["contrastive"] = contrastive_format

        return formats

    def _validate_gan_structure(self, extracted_path: Path) -> bool:
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

    def _validate_supervised_structure(self, extracted_path: Path) -> bool:
        """Validate supervised dataset structure."""
        # Try to find train directory with flexible naming
        keywords = ["train", "traing", "trainging", "training"]
        train_dir = find_directory_by_keywords(extracted_path, keywords)
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

    def _validate_contrastive_structure(self, extracted_path: Path) -> bool:
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

    def detect_format(self, extracted_path: Path) -> Optional[str]:
        """
        Detect the format of a dataset.

        Args:
            extracted_path: Path to the extracted dataset

        Returns:
            Format name if detected, None otherwise
        """
        for format_name, format_spec in self.formats.items():
            try:
                if all(rule(extracted_path) for rule in format_spec.validation_rules):
                    self.logger.info(f"Detected format: {format_name}")
                    return format_name
            except Exception as e:
                self.logger.debug(f"Format detection failed for {format_name}: {e}")

        return None


class ImageSizeValidator:
    """Validates and resizes images according to size constraints."""

    def __init__(
        self, max_size: Tuple[int, int] = (224, 224), max_ratio: float = 2.0
    ) -> None:
        """
        Initialize the image size validator.

        Args:
            max_size: Maximum allowed image size (width, height)
            max_ratio: Maximum ratio of image size to max_size before throwing error
        """
        self.max_size = max_size
        self.max_ratio = max_ratio
        self.logger = get_logger("image_validator")

    def validate_image(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate an image for size constraints.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # Check if image is too large
                if (
                    width > self.max_size[0] * self.max_ratio
                    or height > self.max_size[1] * self.max_ratio
                ):
                    error_msg = (
                        f"Image {image_path} is too large ({width}x{height}). "
                        f"Maximum allowed size is {self.max_size[0] * self.max_ratio}x{self.max_size[1] * self.max_ratio}. "
                        f"Please resize the image to {self.max_size[0]}x{self.max_size[1]} or smaller."
                    )
                    return False, error_msg

                return True, None

        except Exception as e:
            return False, f"Failed to validate image {image_path}: {e}"

    def resize_image(
        self, image: Image.Image, target_size: Tuple[int, int] = (224, 224)
    ) -> Image.Image:
        """
        Resize an image to the target size.

        Args:
            image: PIL Image to resize
            target_size: Target size (width, height)

        Returns:
            Resized image
        """
        return image.resize(target_size, Image.Resampling.LANCZOS)


class DynamicDatasetLoader:
    """
    Dynamic dataset loader that can handle custom zip files with different formats.

    This class provides a flexible system for loading datasets from custom zip files
    with support for different formats and automatic format detection.
    """

    def __init__(self, max_image_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initialize the dynamic dataset loader.

        Args:
            max_image_size: Maximum allowed image size
        """
        self.max_image_size = max_image_size
        self.logger = get_logger("dynamic_loader")
        self.format_detector = DatasetFormatDetector()
        self.image_validator = ImageSizeValidator(max_image_size)

    def load_dataset(
        self,
        zip_path: Union[str, Path],
        task_type: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        **kwargs: Any,
    ) -> Tuple[Dataset[Any], Optional[Dataset[Any]]]:
        """
        Load a dataset from a zip file.

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
        extracted_path = self._extract_zip(zip_path)

        # Detect format if not specified
        if task_type is None:
            task_type = self.format_detector.detect_format(extracted_path)
            if task_type is None:
                raise ValueError(
                    f"Could not detect dataset format. Available formats: {list(self.format_detector.formats.keys())}"
                )

        # Validate images
        self._validate_images(extracted_path)

        # Create dataset based on format
        if task_type == "gan":
            return self._create_gan_dataset(extracted_path, transform, **kwargs)
        elif task_type == "supervised":
            return self._create_supervised_dataset(extracted_path, transform, **kwargs)
        elif task_type == "contrastive":
            return self._create_contrastive_dataset(extracted_path, transform, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _extract_zip(self, zip_path: Path) -> Path:
        """Extract a zip file to a temporary directory."""
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="refrakt_dataset_"))

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find the actual dataset directory (handle nested structures)
        extracted_path = self._find_dataset_directory(temp_dir)
        self.logger.info(f"Extracted dataset to: {extracted_path}")

        return extracted_path

    def _find_dataset_directory(self, temp_dir: Path) -> Path:
        """Find the actual dataset directory within the extracted structure."""
        # Look for common dataset directory names
        dataset_names = ["data", "dataset", "images", "train", "lr"]

        for name in dataset_names:
            potential_path = temp_dir / name
            if potential_path.exists():
                return potential_path

        # If no common name found, return the temp_dir itself
        return temp_dir

    def _find_train_directory(self, extracted_path: Path) -> Optional[Path]:
        """Find a directory that contains 'train' in its name."""
        for item in extracted_path.iterdir():
            if item.is_dir() and "train" in item.name.lower():
                return item
        return None

    def _find_val_directory(self, extracted_path: Path) -> Optional[Path]:
        """Find a directory that contains 'val' or 'test' in its name."""
        for item in extracted_path.iterdir():
            if item.is_dir() and (
                "val" in item.name.lower() or "test" in item.name.lower()
            ):
                return item
        return None

    def _validate_images(self, dataset_path: Path) -> None:
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
            is_valid, error_msg = self.image_validator.validate_image(image_path)
            if not is_valid:
                invalid_images.append((image_path, error_msg))

        if invalid_images:
            error_messages = "\n".join(
                [f"{path}: {msg}" for path, msg in invalid_images[:5]]
            )
            if len(invalid_images) > 5:
                error_messages += f"\n... and {len(invalid_images) - 5} more images"

            raise ValueError(f"Found invalid images:\n{error_messages}")

    def _create_gan_dataset(
        self, dataset_path: Path, transform: Optional[transforms.Compose], **kwargs: Any
    ) -> Tuple[Dataset[Any], Optional[Dataset[Any]]]:
        """Create a GAN dataset from the extracted files."""
        from refrakt_core.datasets import SuperResolutionDataset

        lr_dir = dataset_path / "lr"
        hr_dir = dataset_path / "hr"

        if not lr_dir.exists() or not hr_dir.exists():
            raise ValueError("GAN dataset must have 'lr' and 'hr' directories")

        # Create train dataset
        train_dataset = SuperResolutionDataset(
            lr_dir=lr_dir, hr_dir=hr_dir, transform=transform, train=True
        )

        # Create validation dataset if val directory exists
        val_dataset = None
        val_lr_dir = dataset_path / "val" / "lr"
        val_hr_dir = dataset_path / "val" / "hr"

        if val_lr_dir.exists() and val_hr_dir.exists():
            val_dataset = SuperResolutionDataset(
                lr_dir=val_lr_dir, hr_dir=val_hr_dir, transform=transform, train=False
            )

        return train_dataset, val_dataset

    def _create_supervised_dataset(
        self, dataset_path: Path, transform: Optional[transforms.Compose], **kwargs: Any
    ) -> Tuple[Dataset[Any], Optional[Dataset[Any]]]:
        """Create a supervised dataset from the extracted files."""
        from torchvision.datasets import ImageFolder

        # Find train directory with flexible naming
        train_dir = self._find_train_directory(dataset_path)
        if train_dir is None:
            # Fallback to exact "train" directory
            train_dir = dataset_path / "train"

        if not train_dir.exists():
            raise ValueError(
                "Supervised dataset must have a directory containing 'train' in its name"
            )

        # Find validation directory with flexible naming
        val_dir = self._find_val_directory(dataset_path)
        if val_dir is None:
            # Fallback to exact "val" directory
            val_dir = dataset_path / "val"

        # Create train dataset
        train_dataset = ImageFolder(root=train_dir, transform=transform)

        # Create validation dataset if val directory exists
        val_dataset = None
        if val_dir.exists():
            val_dataset = ImageFolder(root=val_dir, transform=transform)

        return train_dataset, val_dataset

    def _create_contrastive_dataset(
        self, dataset_path: Path, transform: Optional[transforms.Compose], **kwargs: Any
    ) -> Tuple[Dataset[Any], Optional[Dataset[Any]]]:
        """Create a contrastive dataset from the extracted files."""
        from refrakt_core.datasets import ContrastiveDataset

        images_dir = dataset_path / "images"

        if not images_dir.exists():
            raise ValueError("Contrastive dataset must have an 'images' directory")

        # Create a simple dataset that returns image paths
        class ImagePathDataset(Dataset[Path]):
            def __init__(self, images_dir: Path) -> None:
                self.images_dir = images_dir
                self.image_files: List[Path] = (
                    list(images_dir.glob("*.png"))
                    + list(images_dir.glob("*.jpg"))
                    + list(images_dir.glob("*.jpeg"))
                )

            def __len__(self) -> int:
                return len(self.image_files)

            def __getitem__(self, idx: int) -> Path:
                return self.image_files[idx]

        base_dataset = ImagePathDataset(images_dir)

        # Create train dataset
        train_dataset = ContrastiveDataset(
            base_dataset=base_dataset, transform=transform, train=True
        )

        # Create validation dataset if val directory exists
        val_dataset = None
        val_dir = dataset_path / "val"

        if val_dir.exists():
            val_base_dataset = ImagePathDataset(val_dir)
            val_dataset = ContrastiveDataset(
                base_dataset=val_base_dataset, transform=transform, train=False
            )

        return train_dataset, val_dataset

    def create_dataloader(
        self,
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
