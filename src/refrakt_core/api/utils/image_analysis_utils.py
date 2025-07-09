"""
Image analysis utilities for Refrakt.

This module provides comprehensive utility functions for image analysis and resizing,
extracted from the main train_utils.py to reduce complexity and improve maintainability.

The module handles:
- Image size analysis and statistics calculation
- Dataset sampling for efficient analysis
- Image resizing with aspect ratio preservation
- Resized dataset wrapper creation
- Size boundary validation and checking
- Smart image resizing with validation bypass

These utilities ensure robust image analysis operations with proper error handling,
automatic size optimization, and comprehensive dataset processing capabilities.

Typical usage involves calling these utility functions to analyze and optimize
dataset images for training and inference operations.
"""

from typing import Any, List, Tuple

import torch
import torchvision  # type: ignore[import-untyped]

from refrakt_core.api.helpers.image_analysis_helpers import (
    _analyze_sample_sizes,
    _sample_dataset_indices,
)
from refrakt_core.resizers.image_resizer import ImageSizeConfig, SmartImageResizer


def analyze_image_sizes(
    dataset: Any,
    max_size: Tuple[int, int] = (448, 448),
    min_size: Tuple[int, int] = (32, 32),
    sample_count: int = 100,
) -> Tuple[List[Tuple[int, int]], bool, int, int]:
    """
    Analyze image sizes in dataset.

    Args:
        dataset: The dataset to analyze
        max_size: Maximum allowed image size
        min_size: Minimum allowed image size
        sample_count: Number of images to sample

    Returns:
        Tuple of (sizes, needs_resize, oversized_count, undersized_count)
    """
    if hasattr(dataset, "__len__") and len(dataset) == 0:
        return ([], False, 0, 0)
    # Sample images to analyze sizes
    sample_indices = _sample_dataset_indices(len(dataset), sample_count)
    return _analyze_sample_sizes(dataset, sample_indices, max_size, min_size)


def calculate_size_statistics(
    sizes: List[Tuple[int, int]],
) -> Tuple[float, float, int, int, int, int]:
    """
    Calculate statistics from image sizes.

    Args:
        sizes: List of image sizes

    Returns:
        Tuple of (avg_width, avg_height, max_width, max_height, min_width, min_height)
    """
    if not sizes:
        raise ValueError("Empty sizes list")
    if not all(isinstance(s, tuple) and len(s) == 2 for s in sizes):
        raise TypeError("All sizes must be tuples of length 2")
    avg_width = sum(s[0] for s in sizes) / len(sizes)
    avg_height = sum(s[1] for s in sizes) / len(sizes)
    max_width = max(s[0] for s in sizes)
    max_height = max(s[1] for s in sizes)
    min_width = min(s[0] for s in sizes)
    min_height = min(s[1] for s in sizes)

    return avg_width, avg_height, max_width, max_height, min_width, min_height


def create_resized_dataset(dataset: Any, target_size: Tuple[int, int]) -> Any:
    """
    Create a resized dataset wrapper.

    Args:
        dataset: Original dataset
        target_size: Target size for resizing

    Returns:
        Resized dataset wrapper
    """
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise TypeError("target_size must be a tuple of length 2")
    if not all(isinstance(x, int) for x in target_size):
        raise ValueError("target_size must contain integers")
    # Initialize size validator and resizer
    size_config = ImageSizeConfig(
        standard_size=target_size, max_size=(448, 448), min_size=(28, 28)
    )
    resizer = SmartImageResizer(size_config)

    # Create a wrapper dataset that resizes images on-the-fly
    class ResizedDataset:
        def __init__(
            self, original_dataset: Any, resizer: Any, target_size: Tuple[int, int]
        ) -> None:
            self.original_dataset = original_dataset
            self.resizer = resizer
            self.target_size = target_size

        def __len__(self) -> int:
            return len(self.original_dataset)

        def __getitem__(self, idx: int) -> Any:
            sample = self.original_dataset[idx]

            if isinstance(sample, (tuple, list)):
                # Handle (image, label) format
                image, *rest = sample
                resized_image = self._resize_image(image)
                return (resized_image, *rest)
            elif isinstance(sample, dict):
                # Handle dict format
                resized_sample = {}
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor) and value.dim() >= 2:
                        resized_sample[key] = self._resize_image(value)
                    else:
                        resized_sample[key] = value
                return resized_sample
            else:
                # Handle single image
                return self._resize_image(sample)

        def _resize_image(self, image: Any) -> Any:
            """Resize image using SmartImageResizer"""
            if isinstance(image, torch.Tensor):
                # Convert tensor to PIL for resizing
                if image.dim() == 3:  # (C, H, W)
                    pil_image = torchvision.transforms.ToPILImage()(image)
                else:  # (H, W)
                    pil_image = torchvision.transforms.ToPILImage()(image.unsqueeze(0))

                # Resize using SmartImageResizer's internal method to bypass validation
                resized_pil = self.resizer._resize_maintain_aspect(
                    pil_image, self.target_size
                )

                # Convert back to tensor
                return torchvision.transforms.ToTensor()(resized_pil)
            else:
                # PIL Image - use internal method to bypass validation
                return self.resizer._resize_maintain_aspect(image, self.target_size)

    return ResizedDataset(dataset, resizer, target_size)
