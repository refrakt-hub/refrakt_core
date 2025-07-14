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


if __name__ == "__main__":
    pytest.main([__file__])
