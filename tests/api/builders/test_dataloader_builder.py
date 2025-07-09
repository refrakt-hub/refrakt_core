import pytest
import torch
from omegaconf import OmegaConf

from src.refrakt_core.api.builders.dataloader_builder import build_dataloader


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=10):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


@pytest.fixture
def dataset():
    return DummyDataset(5)


@pytest.fixture
def base_cfg():
    return OmegaConf.create(
        {
            "params": {
                "batch_size": 2,
                "shuffle": True,
                "num_workers": 0,
                "drop_last": False,
            }
        }
    )


class TestDataloaderBuilder:
    # Smoke Tests
    def test_build_dataloader_smoke(self, dataset, base_cfg):
        loader = build_dataloader(dataset, base_cfg)
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size == 2
        assert hasattr(loader, "__iter__")

    def test_build_dataloader_no_params_key_smoke(self, dataset):
        cfg = OmegaConf.create({"batch_size": 3, "shuffle": False})
        loader = build_dataloader(dataset, cfg)
        assert loader.batch_size == 3
        assert hasattr(loader, "__iter__")

    # Sanity Tests
    def test_build_dataloader_sanity_drop_last(self, dataset, base_cfg):
        base_cfg.params["drop_last"] = True
        loader = build_dataloader(dataset, base_cfg)
        data = list(loader)
        assert len(data) == 2  # 2 batches, last dropped

    def test_build_dataloader_sanity_num_workers(self, dataset, base_cfg):
        base_cfg.params["num_workers"] = 2
        loader = build_dataloader(dataset, base_cfg)
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_build_dataloader_sanity_iterate(self, dataset, base_cfg):
        loader = build_dataloader(dataset, base_cfg)
        data = list(loader)
        assert all(
            isinstance(x, int)
            or (isinstance(x, torch.Tensor) and x.dtype == torch.int64)
            for batch in data
            for x in (batch if isinstance(batch, (list, tuple)) else [batch])
        )

    # Unit Tests
    def test_build_dataloader_unit_missing_batch_size(self, dataset):
        cfg = OmegaConf.create({"params": {"shuffle": True}})
        with pytest.raises(KeyError):
            build_dataloader(dataset, cfg)

    def test_build_dataloader_unit_params_not_dict(self, dataset):
        cfg = OmegaConf.create({"params": 123})
        with pytest.raises(TypeError):
            build_dataloader(dataset, cfg)

    def test_build_dataloader_unit_default_values(self, dataset):
        cfg = OmegaConf.create({"params": {"batch_size": 1}})
        loader = build_dataloader(dataset, cfg)
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size == 1
        assert loader.drop_last is False
        assert loader.num_workers == 0
        # Can't assert loader.shuffle (not a public attribute)

    def test_build_dataloader_unit_params_key_missing(self, dataset):
        cfg = OmegaConf.create({"shuffle": True, "batch_size": 2})
        loader = build_dataloader(dataset, cfg)
        assert loader.batch_size == 2
        # Can't assert loader.shuffle (not a public attribute)

    def test_build_dataloader_unit_batch_size_zero(self, dataset):
        cfg = OmegaConf.create({"params": {"batch_size": 0, "shuffle": True}})
        with pytest.raises(ValueError):
            build_dataloader(dataset, cfg)
