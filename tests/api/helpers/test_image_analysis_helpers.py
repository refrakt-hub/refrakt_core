import importlib
from types import SimpleNamespace

import pytest
import torch

import refrakt_core.api.helpers.image_analysis_helpers as img_helpers


class DummyDataset:
    def __getitem__(self, idx):
        # Returns (image, label) tuple
        return (torch.ones(3, 32, 32), 1)

    def __len__(self):
        return 10


class TestImageAnalysisHelpers:
    # Smoke Test
    def test_import_image_analysis_helpers(self):
        importlib.reload(img_helpers)

    # Sanity Tests
    def test_extract_image_from_sample_tuple(self):
        sample = (torch.ones(3, 32, 32), 1)
        img = img_helpers._extract_image_from_sample(sample)
        assert isinstance(img, torch.Tensor)

    def test_extract_image_from_sample_dict(self):
        sample = {"lr": torch.ones(3, 32, 32), "hr": torch.ones(3, 32, 32)}
        img = img_helpers._extract_image_from_sample(sample)
        assert isinstance(img, torch.Tensor)

    def test_extract_image_from_sample_direct(self):
        sample = torch.ones(3, 32, 32)
        img = img_helpers._extract_image_from_sample(sample)
        assert isinstance(img, torch.Tensor)

    def test_get_image_size_tensor_3d(self):
        img = torch.ones(3, 32, 64)
        w, h = img_helpers._get_image_size(img)
        assert (w, h) == (64, 32)

    def test_get_image_size_tensor_2d(self):
        img = torch.ones(32, 64)
        w, h = img_helpers._get_image_size(img)
        assert (w, h) == (64, 32)

    def test_check_size_bounds(self):
        size = (32, 32)
        max_size = (64, 64)
        min_size = (16, 16)
        needs_resize, is_oversized, is_undersized = img_helpers._check_size_bounds(
            size, max_size, min_size
        )
        assert needs_resize is False
        assert is_oversized is False
        assert is_undersized is False

    def test_check_size_bounds_oversized(self):
        size = (128, 128)
        max_size = (64, 64)
        min_size = (16, 16)
        needs_resize, is_oversized, is_undersized = img_helpers._check_size_bounds(
            size, max_size, min_size
        )
        assert needs_resize is True
        assert is_oversized is True
        assert is_undersized is False

    def test_check_size_bounds_undersized(self):
        size = (8, 8)
        max_size = (64, 64)
        min_size = (16, 16)
        needs_resize, is_oversized, is_undersized = img_helpers._check_size_bounds(
            size, max_size, min_size
        )
        assert needs_resize is True
        assert is_oversized is False
        assert is_undersized is True

    def test_sample_dataset_indices(self):
        indices = img_helpers._sample_dataset_indices(10, 3)
        assert len(indices) == 3
        assert indices[0] == 0

    def test_sample_dataset_indices_zero(self):
        indices = img_helpers._sample_dataset_indices(0, 3)
        assert indices == []

    def test_analyze_sample_sizes(self):
        dataset = DummyDataset()
        indices = [0, 1, 2]
        max_size = (64, 64)
        min_size = (16, 16)
        sizes, needs_resize, oversized_count, undersized_count = (
            img_helpers._analyze_sample_sizes(dataset, indices, max_size, min_size)
        )
        assert all(isinstance(s, tuple) for s in sizes)
        assert needs_resize is False
        assert oversized_count == 0
        assert undersized_count == 0
