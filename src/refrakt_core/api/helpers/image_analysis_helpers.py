"""
Image analysis helper functions for Refrakt.

This module contains internal helper functions used by the image analysis utilities.
"""

import torch
from typing import Any, Tuple


def _extract_image_from_sample(sample: Any) -> Any:
    """Extract image from dataset sample."""
    if isinstance(sample, (tuple, list)):
        # Handle (image, label) format
        return sample[0]
    elif isinstance(sample, dict):
        # Handle dict format (e.g., {'lr': tensor, 'hr': tensor})
        return list(sample.values())[0]
    else:
        return sample


def _get_image_size(image: Any) -> Tuple[int, int]:
    """Get image size from tensor or PIL image."""
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:  # (C, H, W)
            return (image.size(2), image.size(1))  # (W, H)
        else:  # (H, W)
            return (image.size(1), image.size(0))  # (W, H)
    else:
        return image.size  # PIL Image


def _check_size_bounds(size: Tuple[int, int], max_size: Tuple[int, int], min_size: Tuple[int, int]) -> Tuple[bool, bool, bool]:
    """Check if image size is within bounds."""
    width, height = size
    is_oversized = width > max_size[0] or height > max_size[1]
    is_undersized = width < min_size[0] or height < min_size[1]
    needs_resize = is_oversized or is_undersized
    return needs_resize, is_oversized, is_undersized


def _sample_dataset_indices(dataset_length: int, sample_count: int) -> list:
    """Generate sample indices for dataset analysis."""
    sample_count = min(sample_count, dataset_length)
    return list(range(0, dataset_length, max(1, dataset_length // sample_count)))[:sample_count]


def _analyze_sample_sizes(dataset: Any, sample_indices: list, max_size: Tuple[int, int], min_size: Tuple[int, int]) -> Tuple[list, bool, int, int]:
    """Analyze sizes of sampled images."""
    sizes = []
    needs_resize = False
    oversized_count = 0
    undersized_count = 0

    for idx in sample_indices:
        try:
            # Get image from dataset
            sample = dataset[idx]
            image = _extract_image_from_sample(sample)
            size = _get_image_size(image)
            sizes.append(size)

            # Check if size is outside acceptable range
            needs_resize_sample, is_oversized, is_undersized = _check_size_bounds(size, max_size, min_size)
            
            if is_oversized:
                oversized_count += 1
            if is_undersized:
                undersized_count += 1
            if needs_resize_sample:
                needs_resize = True

        except Exception:
            continue

    return sizes, needs_resize, oversized_count, undersized_count 