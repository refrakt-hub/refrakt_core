"""
Tests for the standard transforms system.

This module contains comprehensive tests for the standard transforms system
including smoke tests, sanity checks, and unit tests.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image
import torchvision.transforms as T

from refrakt_core.resizers.standard_transforms import (
    ImageSizeConfig,
    validate_image_size,
    resize_image_maintain_aspect,
    resize_image_crop,
    resize_image_stretch,
    create_standard_transform,
    create_classification_transform,
    create_contrastive_transform,
    create_gan_transform,
    validate_transform_input,
    StandardImageTransform
)

# Smoke Tests
def test_image_size_config_smoke():
    """Smoke test: Create default image size config."""
    config = ImageSizeConfig()
    assert config.standard_size == (224, 224)
    assert config.max_size == (448, 448)
    assert config.min_size == (28, 28)
    assert config.aspect_ratio_tolerance == 0.1
    assert config.interpolation_method == "lanczos"


def test_validate_image_size_smoke():
    """Smoke test: Validate valid image size."""
    size = (224, 224)
    is_valid, error_msg = validate_image_size(size)
    assert is_valid is True
    assert error_msg is None


def test_resize_image_maintain_aspect_smoke():
    """Smoke test: Resize image maintaining aspect ratio."""
    img = Image.new('RGB', (200, 100), color='red')
    target_size = (224, 224)
    resized = resize_image_maintain_aspect(img, target_size)
    assert resized.size == target_size
    assert resized.mode == img.mode


def test_create_standard_transform_smoke():
    """Smoke test: Create basic standard transform."""
    transform = create_standard_transform()
    assert isinstance(transform, T.Compose)
    assert len(transform.transforms) > 0

# Sanity Tests
def test_image_size_config_sanity():
    """Sanity test: Create custom image size config."""
    config = ImageSizeConfig(
        standard_size=(512, 512),
        max_size=(1024, 1024),
        min_size=(64, 64),
        aspect_ratio_tolerance=0.2,
        interpolation_method="bilinear"
    )
    assert config.standard_size == (512, 512)
    assert config.max_size == (1024, 1024)
    assert config.min_size == (64, 64)
    assert config.aspect_ratio_tolerance == 0.2
    assert config.interpolation_method == "bilinear"


def test_validate_image_size_sanity():
    """Sanity test: Validate image size boundaries."""
    # Too small
    size = (16, 16)
    is_valid, error_msg = validate_image_size(size, min_size=(28, 28))
    assert is_valid is False
    assert error_msg is not None
    assert "too small" in error_msg
    # Too large
    size = (1000, 1000)
    is_valid, error_msg = validate_image_size(size, max_size=(448, 448))
    assert is_valid is False
    assert error_msg is not None
    assert "too large" in error_msg


def test_resize_image_crop_sanity():
    """Sanity test: Resize image with cropping."""
    img = Image.new('RGB', (100, 200), color='blue')
    target_size = (224, 224)
    resized = resize_image_crop(img, target_size)
    assert resized.size == target_size
    assert resized.mode == img.mode


def test_create_standard_transform_with_options_sanity():
    """Sanity test: Create standard transform with options."""
    transform = create_standard_transform(
        target_size=(512, 512),
        resize_strategy="crop",
        normalize=False,
        augment=True
    )
    assert isinstance(transform, T.Compose)
    assert len(transform.transforms) > 0

# Unit Tests
def test_validate_image_size_with_path_unit():
    """Unit test: Validate image size with path."""
    size = (1000, 1000)
    image_path = Path("/path/to/image.png")
    is_valid, error_msg = validate_image_size(size, image_path, max_size=(448, 448))
    assert is_valid is False
    assert error_msg is not None
    assert str(image_path) in error_msg


def test_resize_image_stretch_unit():
    """Unit test: Resize image with stretching."""
    img = Image.new('RGB', (100, 200), color='green')
    target_size = (224, 224)
    resized = resize_image_stretch(img, target_size)
    assert resized.size == target_size
    assert resized.mode == img.mode


def test_standard_image_transform_call_unit():
    """Unit test: StandardImageTransform __call__ with PIL image."""
    img = Image.new('RGB', (224, 224), color='red')
    transform = StandardImageTransform()
    result = transform(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape[1:] == (224, 224) 