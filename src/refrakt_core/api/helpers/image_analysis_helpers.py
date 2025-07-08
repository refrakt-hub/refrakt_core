"""
Image analysis helper functions for Refrakt.

This module contains internal helper functions used by the image analysis utilities
to handle image extraction, size analysis, and dataset sampling operations.

The module handles:
- Image extraction from various dataset sample formats
- Image size determination for different data types
- Size boundary checking and validation
- Dataset sampling for efficient analysis
- Batch image size analysis with error handling

These utilities ensure robust image analysis operations with proper error handling
and support for various image formats and dataset structures.

Typical usage involves calling these helper functions from the main image analysis
utilities to perform dataset analysis and image processing operations.
"""

from typing import Any, List, Tuple, Union, cast

import torch


def _extract_image_from_sample(sample: Any) -> Union[torch.Tensor, Any]:
    """
    Extract image from dataset sample.

    This function handles various dataset sample formats and extracts
    the image component for analysis. It supports tuple, list, and
    dictionary formats commonly used in datasets.

    Args:
        sample: Dataset sample that may be a tuple, list, dict, or direct image

    Returns:
        Extracted image for analysis

    Note:
        This function handles common dataset formats:
        - (image, label) tuples
        - [image, label] lists
        - {'lr': tensor, 'hr': tensor} dictionaries
        - Direct image objects
    """
    if isinstance(sample, (tuple, list)):
        # Handle (image, label) format
        return sample[0]
    elif isinstance(sample, dict):
        # Handle dict format (e.g., {'lr': tensor, 'hr': tensor})
        return list(sample.values())[0]
    else:
        return sample


def _get_image_size(image: Any) -> Tuple[int, int]:
    """
    Get image size from tensor or PIL image.

    This function extracts the width and height dimensions from various
    image formats, handling both PyTorch tensors and PIL images.

    Args:
        image: Image object (PyTorch tensor or PIL image)

    Returns:
        Tuple of (width, height) dimensions

    Note:
        For PyTorch tensors, this function handles both (C, H, W) and (H, W)
        formats, returning dimensions in (width, height) order.
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:  # (C, H, W)
            return (image.size(2), image.size(1))  # (W, H)
        else:  # (H, W)
            return (image.size(1), image.size(0))  # (W, H)
    else:
        return cast(Tuple[int, int], image.size)  # PIL Image


def _check_size_bounds(
    size: Tuple[int, int], max_size: Tuple[int, int], min_size: Tuple[int, int]
) -> Tuple[bool, bool, bool]:
    """
    Check if image size is within bounds.

    This function validates whether an image size falls within acceptable
    minimum and maximum bounds, providing detailed status information.

    Args:
        size: Image size as (width, height) tuple
        max_size: Maximum acceptable size as (width, height) tuple
        min_size: Minimum acceptable size as (width, height) tuple

    Returns:
        Tuple containing:
        - needs_resize: Whether the image needs resizing
        - is_oversized: Whether the image exceeds maximum bounds
        - is_undersized: Whether the image is below minimum bounds
    """
    width, height = size
    is_oversized = width > max_size[0] or height > max_size[1]
    is_undersized = width < min_size[0] or height < min_size[1]
    needs_resize = is_oversized or is_undersized
    return needs_resize, is_oversized, is_undersized


def _sample_dataset_indices(dataset_length: int, sample_count: int) -> List[int]:
    """
    Generate sample indices for dataset analysis.

    This function creates a list of indices to sample from a dataset
    for efficient analysis, ensuring even distribution across the dataset.

    Args:
        dataset_length: Total number of samples in the dataset
        sample_count: Number of samples to analyze

    Returns:
        List of indices to sample from the dataset

    Note:
        The function ensures the sample count doesn't exceed the dataset length
        and provides even distribution across the dataset for representative analysis.
    """
    if dataset_length == 0 or sample_count == 0:
        return []
    sample_count = min(sample_count, dataset_length)
    return list(range(0, dataset_length, max(1, dataset_length // sample_count)))[
        :sample_count
    ]


def _analyze_sample_sizes(
    dataset: Any,
    sample_indices: List[int],
    max_size: Tuple[int, int],
    min_size: Tuple[int, int],
) -> Tuple[List[Tuple[int, int]], bool, int, int]:
    """
    Analyze sizes of sampled images.

    This function analyzes a sample of images from a dataset to determine
    size statistics and identify images that need resizing.

    Args:
        dataset: Dataset to analyze
        sample_indices: List of indices to sample from the dataset
        max_size: Maximum acceptable image size
        min_size: Minimum acceptable image size

    Returns:
        Tuple containing:
        - sizes: List of (width, height) tuples for sampled images
        - needs_resize: Whether any images need resizing
        - oversized_count: Number of images exceeding maximum size
        - undersized_count: Number of images below minimum size

    Note:
        This function includes error handling for individual samples,
        allowing analysis to continue even if some images cannot be processed.
    """
    sizes: List[Tuple[int, int]] = []
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
            needs_resize_sample, is_oversized, is_undersized = _check_size_bounds(
                size, max_size, min_size
            )

            if is_oversized:
                oversized_count += 1
            if is_undersized:
                undersized_count += 1
            if needs_resize_sample:
                needs_resize = True

        except Exception:
            continue

    return sizes, needs_resize, oversized_count, undersized_count
