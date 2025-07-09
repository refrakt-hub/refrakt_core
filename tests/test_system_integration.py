"""
Integration tests for the Refrakt system.

This module contains integration tests that verify all the new systems
work together properly, including registry, logging, hyperparameter overrides,
and dataset loading.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf

from refrakt_core.hooks.hyperparameter_override import (
    apply_overrides,
    extract_overrides_from_args,
)
from refrakt_core.loaders.dataset_loader import load_dataset
from refrakt_core.logging_config import (
    configure_logger,
    get_logger,
    get_logging_manager,
)
from refrakt_core.registry.safe_registry import (
    get_dataset,
    get_model,
    get_registry,
    register_dataset,
    register_model,
)
from refrakt_core.resizers.standard_transforms import create_standard_transform


# Smoke Tests
def test_registry_and_logging_integration_smoke():
    """Smoke test: Registry and logging work together."""
    logger = configure_logger("test_integration", console=True, debug=True)

    @register_model("test_model")
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    model_cls = get_model("test_model")
    model = model_cls()
    logger.info(f"Registered and retrieved model: {type(model).__name__}")
    assert model_cls == TestModel
    assert isinstance(model, torch.nn.Module)
    get_registry().clear("models")


# Sanity Tests
def test_hyperparameter_override_integration_sanity():
    """Sanity test: Hyperparameter overrides work with config."""
    base_config = {
        "model": {"name": "default", "params": {"lr": 0.001, "batch_size": 32}},
        "training": {"epochs": 10},
    }
    overrides = ["model.name=ResNet", "model.params.lr=0.0005", "training.epochs=20"]
    base_config = OmegaConf.create(base_config)
    result = apply_overrides(base_config, overrides)
    assert result["model"]["name"] == "ResNet"
    assert result["model"]["params"]["lr"] == 0.0005
    assert result["training"]["epochs"] == 20
    assert result["model"]["params"]["batch_size"] == 32


def test_command_line_override_extraction_sanity():
    """Sanity test: Extract overrides from command-line arguments."""
    args = [
        "python",
        "-m",
        "refrakt_core.api.train",
        "--config",
        "config.yaml",
        "config.model.name=ResNet",
        "config.optimizer.lr=0.0005",
        "config.trainer.epochs=20",
        "--device",
        "cuda",
    ]
    overrides, remaining = extract_overrides_from_args(args)
    assert overrides == [
        "config.model.name=ResNet",
        "config.optimizer.lr=0.0005",
        "config.trainer.epochs=20",
    ]
    assert remaining == [
        "python",
        "-m",
        "refrakt_core.api.train",
        "--config",
        "config.yaml",
        "--device",
        "cuda",
    ]


# Unit Tests
def test_transform_and_dataset_integration_unit():
    """Unit test: Transforms and dataset loading work together."""
    transform = create_standard_transform(
        target_size=(64, 64), normalize=True, augment=False
    )
    with patch(
        "refrakt_core.loaders.dataset_loader.load_torchvision_dataset"
    ) as mock_load:
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(return_value=("data", "label"))
        mock_load.return_value = mock_dataset
        train_dataset, val_dataset = load_dataset("mnist", transform=transform)
        mock_load.assert_called()
        assert train_dataset is not None
        assert val_dataset is not None


def test_complete_pipeline_integration_unit():
    """Unit test: Complete pipeline from config to model."""
    config = {
        "model": {
            "name": "test_model",
            "params": {"input_size": 784, "output_size": 10},
        },
        "training": {"epochs": 10, "lr": 0.001},
    }
    overrides = [
        "model.params.input_size=1024",
        "training.epochs=20",
        "training.lr=0.0005",
    ]
    config = OmegaConf.create(config)
    config_with_overrides = apply_overrides(config, overrides)
    assert config_with_overrides["model"]["params"]["input_size"] == 1024
    assert config_with_overrides["training"]["epochs"] == 20
    assert config_with_overrides["training"]["lr"] == 0.0005

    @register_model("test_model")
    class TestModel(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.linear = torch.nn.Linear(input_size, output_size)

        def forward(self, x):
            return self.linear(x)

    model_cls = get_model("test_model")
    model_params = config_with_overrides["model"]["params"]
    model = model_cls(**model_params)
    assert model.linear.in_features == 1024
    assert model.linear.out_features == 10


if __name__ == "__main__":
    pytest.main([__file__])
