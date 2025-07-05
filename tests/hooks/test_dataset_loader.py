"""
Tests for the dataset loader system.

This module contains comprehensive tests for the dataset loader system
including smoke tests, sanity checks, and unit tests.
"""

import pytest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from refrakt_core.loaders.dataset_loader import (
    validate_gan_structure,
    validate_supervised_structure,
    validate_contrastive_structure,
    detect_dataset_format,
    validate_image_size,
    extract_zip_file,
    _find_dataset_directory,
    validate_dataset_images,
    load_custom_dataset,
    load_torchvision_dataset,
    create_dataloader,
    load_dataset
)


# Smoke Tests
def test_validate_gan_structure_smoke():
    """Smoke test: Validate valid GAN structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create GAN structure
        lr_dir = temp_path / "lr"
        hr_dir = temp_path / "hr"
        lr_dir.mkdir()
        hr_dir.mkdir()
        
        # Create matching files
        img1 = Image.new('RGB', (64, 64), color='red')
        img1.save(lr_dir / "image1.png")
        img2 = Image.new('RGB', (128, 128), color='blue')
        img2.save(hr_dir / "image1.png")
        
        assert validate_gan_structure(temp_path) is True


def test_validate_supervised_structure_smoke():
    """Smoke test: Validate valid supervised structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create supervised structure
        train_dir = temp_path / "train"
        train_dir.mkdir()
        
        class1_dir = train_dir / "cat"
        class2_dir = train_dir / "dog"
        class1_dir.mkdir()
        class2_dir.mkdir()
        
        # Create images in class directories
        img1 = Image.new('RGB', (64, 64), color='red')
        img1.save(class1_dir / "cat1.png")
        img2 = Image.new('RGB', (64, 64), color='blue')
        img2.save(class2_dir / "dog1.png")
        
        assert validate_supervised_structure(temp_path) is True


def test_validate_contrastive_structure_smoke():
    """Smoke test: Validate valid contrastive structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create contrastive structure
        images_dir = temp_path / "images"
        images_dir.mkdir()
        
        # Create images
        img1 = Image.new('RGB', (64, 64), color='red')
        img1.save(images_dir / "image1.png")
        img2 = Image.new('RGB', (64, 64), color='blue')
        img2.save(images_dir / "image2.jpg")
        
        assert validate_contrastive_structure(temp_path) is True


def test_detect_dataset_format_smoke():
    """Smoke test: Detect dataset format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create GAN structure
        lr_dir = temp_path / "lr"
        hr_dir = temp_path / "hr"
        lr_dir.mkdir()
        hr_dir.mkdir()
        
        (lr_dir / "image1.png").touch()
        (hr_dir / "image1.png").touch()
        
        format_name = detect_dataset_format(temp_path)
        assert format_name == "gan"


def test_validate_image_size_smoke():
    """Smoke test: Validate valid image size."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a test image
        img_path = temp_path / "test.png"
        img = Image.new('RGB', (224, 224), color='red')
        img.save(img_path)
        
        is_valid, error_msg = validate_image_size(img_path)
        
        assert is_valid is True
        assert error_msg is None


def test_extract_zip_file_smoke():
    """Smoke test: Extract zip file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a test zip file
        zip_path = temp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            zip_file.writestr("test.txt", "test content")
        
        # Extract the zip file
        extracted_path = extract_zip_file(zip_path)
        
        # Verify extraction
        assert extracted_path.exists()


def test_load_torchvision_dataset_smoke():
    """Smoke test: Load torchvision dataset."""
    result = load_torchvision_dataset("mnist")
    assert result is not None


def test_create_dataloader_smoke():
    """Smoke test: Create dataloader."""
    # Create a mock dataset
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=100)
    mock_dataset.__getitem__ = Mock(return_value=("data", "label"))
    
    dataloader = create_dataloader(mock_dataset, batch_size=32)
    
    assert dataloader is not None
    assert hasattr(dataloader, '__iter__')


# Sanity Tests
def test_validate_gan_structure_sanity():
    """Sanity test: Validate invalid GAN structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Missing directories
        assert validate_gan_structure(temp_path) is False
        
        # Only lr directory
        lr_dir = temp_path / "lr"
        lr_dir.mkdir()
        assert validate_gan_structure(temp_path) is False
        
        # Only hr directory
        shutil.rmtree(lr_dir)
        hr_dir = temp_path / "hr"
        hr_dir.mkdir()
        assert validate_gan_structure(temp_path) is False


def test_validate_supervised_structure_sanity():
    """Sanity test: Validate invalid supervised structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # No train directory
        assert validate_supervised_structure(temp_path) is False
        
        # Train directory exists but no class directories
        train_dir = temp_path / "train"
        train_dir.mkdir()
        assert validate_supervised_structure(temp_path) is False


def test_validate_contrastive_structure_sanity():
    """Sanity test: Validate invalid contrastive structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # No images directory
        assert validate_contrastive_structure(temp_path) is False
        
        # Images directory with no images
        images_dir = temp_path / "images"
        images_dir.mkdir()
        assert validate_contrastive_structure(temp_path) is False


def test_validate_image_size_sanity():
    """Sanity test: Validate image size boundaries."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Test small image (should be valid)
        small_img_path = temp_path / "small.png"
        small_img = Image.new('RGB', (16, 16), color='red')
        small_img.save(small_img_path)
        is_valid, error_msg = validate_image_size(small_img_path, max_size=(32, 32))
        # 16x16 should be valid with max_size=(32,32) and default max_ratio=2.0
        assert is_valid is True
        # Test too large image
        large_img_path = temp_path / "large.png"
        large_img = Image.new('RGB', (1000, 1000), color='blue')
        large_img.save(large_img_path)
        is_valid, error_msg = validate_image_size(large_img_path, max_size=(448, 448))
        assert is_valid is False
        assert error_msg is not None


def test_detect_dataset_format_sanity():
    """Sanity test: Detect different dataset formats."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test supervised format
        train_dir = temp_path / "train"
        train_dir.mkdir()
        class_dir = train_dir / "class1"
        class_dir.mkdir()
        (class_dir / "image1.png").touch()
        
        format_name = detect_dataset_format(temp_path)
        assert format_name == "supervised"
        
        # Test contrastive format
        shutil.rmtree(train_dir)
        images_dir = temp_path / "images"
        images_dir.mkdir()
        (images_dir / "image1.png").touch()
        
        format_name = detect_dataset_format(temp_path)
        assert format_name == "contrastive"


def test_load_dataset_sanity():
    """Sanity test: Load dataset with auto-detection."""
    with patch('refrakt_core.loaders.dataset_loader.load_custom_dataset') as mock_custom:
        mock_dataset = Mock()
        mock_custom.return_value = mock_dataset
        
        result = load_dataset("test_dataset")
        
        assert result is not None
        mock_custom.assert_called_once()


# Unit Tests
def test_validate_gan_structure_no_matching_files():
    """Unit test: GAN structure with no matching files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        lr_dir = temp_path / "lr"
        hr_dir = temp_path / "hr"
        lr_dir.mkdir()
        hr_dir.mkdir()
        
        # Create non-matching files
        (lr_dir / "image1.png").touch()
        (hr_dir / "image2.png").touch()
        
        assert validate_gan_structure(temp_path) is False


def test_validate_image_size_with_path():
    """Unit test: Validate image size with path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a large test image
        large_img_path = temp_path / "large.png"
        large_img = Image.new('RGB', (1000, 1000), color='red')
        large_img.save(large_img_path)
        
        is_valid, error_msg = validate_image_size(large_img_path, max_size=(448, 448))
        
        assert is_valid is False
        assert error_msg is not None
        assert str(large_img_path) in error_msg


def test_find_dataset_directory():
    """Unit test: Find dataset directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Create nested structure
        nested_dir = temp_path / "data" / "dataset"
        nested_dir.mkdir(parents=True)
        # Create some files
        (nested_dir / "image1.png").touch()
        (nested_dir / "image2.jpg").touch()
        found_dir = _find_dataset_directory(temp_path)
        assert found_dir is not None
        # The function should find the "data" directory, which contains "dataset"
        assert found_dir.name == "data"
        # Check that the dataset subdirectory exists and contains files
        dataset_dir = found_dir / "dataset"
        assert dataset_dir.exists()
        files = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg"))
        assert len(files) >= 2


def test_load_custom_dataset_gan():
    """Unit test: Load custom GAN dataset."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Create GAN structure
        lr_dir = temp_path / "lr"
        hr_dir = temp_path / "hr"
        lr_dir.mkdir()
        hr_dir.mkdir()
        img1 = Image.new('RGB', (64, 64), color='red')
        img1.save(lr_dir / "image1.png")
        img2 = Image.new('RGB', (128, 128), color='blue')
        img2.save(hr_dir / "image1.png")
        # Zip the contents of temp_path (lr and hr directories) at the root
        zip_path = temp_path / "gan_dataset.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for item in temp_path.iterdir():
                if item.is_dir() and item.name != "gan_dataset.zip":
                    for root, dirs, files in os.walk(item):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = str(Path(root).relative_to(temp_path)) + '/' + file
                            zipf.write(file_path, arcname)
        dataset = load_custom_dataset(zip_path, task_type="gan")
        assert dataset is not None
        assert hasattr(dataset, '__len__')
        assert hasattr(dataset, '__getitem__')


def test_load_custom_dataset_supervised():
    """Unit test: Load custom supervised dataset."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Create supervised structure
        train_dir = temp_path / "train"
        train_dir.mkdir()
        class_dir = train_dir / "class1"
        class_dir.mkdir()
        img = Image.new('RGB', (64, 64), color='green')
        img.save(class_dir / "image1.png")
        # Zip the contents of temp_path (train directory) at the root
        zip_path = temp_path / "supervised_dataset.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for item in temp_path.iterdir():
                if item.is_dir() and item.name != "supervised_dataset.zip":
                    for root, dirs, files in os.walk(item):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = str(Path(root).relative_to(temp_path)) + '/' + file
                            zipf.write(file_path, arcname)
        dataset = load_custom_dataset(zip_path, task_type="supervised")
        assert dataset is not None
        assert hasattr(dataset, '__len__')
        assert hasattr(dataset, '__getitem__')


def test_validate_dataset_images_no_images():
    """Unit test: Validate dataset with no images."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create empty directory
        empty_dir = temp_path / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="No image files found in dataset"):
            validate_dataset_images(empty_dir) 