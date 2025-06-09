from unittest.mock import MagicMock, patch

import pytest

from refrakt_core.registry.transform_registry import (TRANSFORM_REGISTRY,
                                                      get_transform,
                                                      register_transform)


def test_torchvision_fallback():
    from torchvision.transforms import RandomHorizontalFlip

    from refrakt_core.registry.transform_registry import get_transform

    t = get_transform("RandomHorizontalFlip", p=0.5)
    assert isinstance(t, RandomHorizontalFlip)


def test_transform_not_found():
    """Test error for unavailable transform"""
    with pytest.raises(ValueError) as excinfo:
        get_transform("invalid_transform")
    assert "Transform 'invalid_transform' not found" in str(excinfo.value)