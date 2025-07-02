import pytest
import torch
from unittest.mock import patch, MagicMock
from refrakt_core.api.core import utils

def test_setup_device_smoke():
    device = utils.setup_device()
    assert device in ("cuda", "cpu")

def test_flatten_and_filter_config_unit():
    nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}, "f": torch.tensor(5)}
    flat = utils.flatten_and_filter_config(nested)
    assert flat["a"] == 1
    assert flat["b.c"] == 2
    assert flat["b.d.e"] == 3
    assert isinstance(flat["f"], torch.Tensor)

@patch("refrakt_core.api.core.utils.build_dataset")
def test_build_datasets_sanity(mock_build_dataset):
    # Mock OmegaConf with .dataset attribute
    class DummyCfg:
        dataset = {"params": {"train": True}}
    mock_build_dataset.side_effect = lambda cfg: cfg
    train, val = utils.build_datasets(DummyCfg())
    assert train["params"]["train"] is True
    assert val["params"]["train"] is False

@patch("refrakt_core.api.core.utils.build_dataloader")
def test_build_dataloaders_sanity(mock_build_dataloader):
    mock_build_dataloader.side_effect = lambda ds, cfg: [ds, cfg]
    train_loader, val_loader = utils.build_dataloaders("train_ds", "val_ds", MagicMock(dataloader={}))
    assert train_loader[0] == "train_ds"
    assert val_loader[0] == "val_ds" 