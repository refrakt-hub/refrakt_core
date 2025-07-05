from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from refrakt_core.api.builders.transform_builder import build_transform


def test_build_transform_simple():
    cfg = [
        {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
        {"name": "RandomVerticalFlip", "params": {"p": 0.5}},
    ]
    with patch(
        "refrakt_core.registry.transform_registry.get_transform"
    ) as get_transform:
        get_transform.side_effect = lambda name, **params: MagicMock(name=name)
        transform = build_transform(cfg)
        assert hasattr(transform, "__call__")


def test_build_transform_nested():
    cfg = [
        {
            "name": "RandomApply",
            "params": {
                "transforms": [{"name": "RandomHorizontalFlip", "params": {}}],
                "p": 0.5,
            },
        }
    ]
    with patch(
        "refrakt_core.registry.transform_registry.get_transform"
    ) as get_transform:
        get_transform.side_effect = lambda name, *args, **params: MagicMock(name=name)
        transform = build_transform(cfg)
        assert hasattr(transform, "__call__")


def test_build_transform_bad_type():
    cfg = "not_a_list"
    with pytest.raises(TypeError):
        build_transform(cfg)
