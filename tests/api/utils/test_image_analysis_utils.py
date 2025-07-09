import importlib
from types import SimpleNamespace

import pytest
import torch

import src.refrakt_core.api.utils.image_analysis_utils as image_analysis_utils


class DummyDataset:
    def __init__(self, n=10, shape=(3, 32, 32)):
        self.n = n
        self.shape = shape

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.ones(*self.shape)


class TestImageAnalysisUtils:
    # Smoke Test
    def test_import_image_analysis_utils(self):
        importlib.reload(image_analysis_utils)

    # Sanity Tests
    def test_analyze_image_sizes_empty(self):
        ds = DummyDataset(n=0)
        sizes, needs_resize, oversized_count, undersized_count = (
            image_analysis_utils.analyze_image_sizes(ds)
        )
        assert sizes == []
        assert needs_resize is False
        assert oversized_count == 0
        assert undersized_count == 0

    def test_analyze_image_sizes_nonempty(self):
        ds = DummyDataset(n=5)
        sizes, needs_resize, oversized_count, undersized_count = (
            image_analysis_utils.analyze_image_sizes(ds, sample_count=2)
        )
        assert isinstance(sizes, list)
        assert isinstance(needs_resize, bool)
        assert isinstance(oversized_count, int)
        assert isinstance(undersized_count, int)

    def test_calculate_size_statistics(self):
        sizes = [(32, 32), (64, 64), (48, 48)]
        stats = image_analysis_utils.calculate_size_statistics(sizes)
        assert len(stats) == 6
        assert stats[0] == pytest.approx(48.0)
        assert stats[1] == pytest.approx(48.0)
        assert stats[2] == 64
        assert stats[3] == 64
        assert stats[4] == 32
        assert stats[5] == 32

    def test_calculate_size_statistics_empty(self):
        with pytest.raises(ValueError):
            image_analysis_utils.calculate_size_statistics([])

    def test_calculate_size_statistics_typeerror(self):
        with pytest.raises(TypeError):
            image_analysis_utils.calculate_size_statistics([(32, 32), (64,)])

    def test_create_resized_dataset_typeerror(self):
        ds = DummyDataset()
        with pytest.raises(TypeError):
            image_analysis_utils.create_resized_dataset(ds, 32)

    def test_create_resized_dataset_valueerror(self):
        ds = DummyDataset()
        with pytest.raises(ValueError):
            image_analysis_utils.create_resized_dataset(ds, (32, "foo"))

    def test_create_resized_dataset_success(self):
        ds = DummyDataset()
        resized_ds = image_analysis_utils.create_resized_dataset(ds, (32, 32))
        assert hasattr(resized_ds, "__getitem__")
        assert hasattr(resized_ds, "__len__")
        sample = resized_ds[0]
        assert sample is not None
