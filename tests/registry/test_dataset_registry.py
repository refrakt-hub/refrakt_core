from unittest.mock import MagicMock, patch

import pytest

from refrakt_core.registry.dataset_registry import (DATASET_REGISTRY,
                                                    get_dataset,
                                                    register_dataset)


# Mock logger
@pytest.fixture
def mock_logger():
    with patch("refrakt_core.logging.get_global_logger") as mock_logger:
        logger = MagicMock()
        mock_logger.return_value = logger
        yield logger

@patch("refrakt_core.registry.dataset_registry.get_global_logger")
def test_register_dataset(mock_get_logger):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    @register_dataset("test_ds")
    class TestDataset:
        pass

    assert "test_ds" in DATASET_REGISTRY
    assert DATASET_REGISTRY["test_ds"] is TestDataset
    mock_logger.debug.assert_called_with("Registering dataset: %s", "test_ds")

@patch("refrakt_core.registry.dataset_registry.get_global_logger")
def test_duplicate_registration(mock_get_logger):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    @register_dataset("duplicate_ds")
    class DS1:
        pass

    @register_dataset("duplicate_ds")
    class DS2:
        pass

    assert DATASET_REGISTRY["duplicate_ds"] is DS1
    mock_logger.debug.assert_called_with(
        "Warning: Dataset '%s' already registered. Skipping.", "duplicate_ds"
    )


def test_get_dataset_success():
    """Test retrieval of registered dataset"""
    @register_dataset("valid_ds")
    class ValidDataset:
        def __init__(self, param):
            self.param = param

    instance = get_dataset("valid_ds", "test_param")
    assert isinstance(instance, ValidDataset)
    assert instance.param == "test_param"
        
def test_get_dataset_torchvision_fallback():
    """Test torchvision dataset fallback"""
    from refrakt_core.registry.dataset_registry import get_dataset
    dataset = get_dataset("FakeData")  # or "CIFAR10", "MNIST"
    assert dataset is not None


def test_get_dataset_not_found():
    """Test handling of unregistered dataset"""
    with pytest.raises(ValueError) as excinfo:
        get_dataset("unknown_ds")
    assert "Dataset 'unknown_ds' not found" in str(excinfo.value)