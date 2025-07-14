import importlib

import pytest
import torch
from omegaconf import DictConfig

from refrakt_core.api.utils import test_utils
from refrakt_core.api.core.logger import RefraktLogger


class DummyLogger(RefraktLogger):
    def info(self, msg):
        self.info_called = True

    def warning(self, msg):
        self.warning_called = True


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False

    def forward(self, x):
        self.called = True
        return x


class TestTestUtils:
    # Smoke Test
    def test_import_test_utils(self):
        importlib.reload(test_utils)

    # Sanity Tests
    def test_load_config_dictconfig(self):
        cfg = DictConfig({"foo": 1})
        out = test_utils._load_config(cfg)
        assert out == cfg

    def test_resolve_model_name_autoencoder(self):
        cfg = DictConfig(
            {
                "model": {"name": "autoencoder", "params": {"variant": "foo"}},
                "dataset": {"params": {}},
            }
        )
        name = test_utils._resolve_model_name(cfg)
        assert name == "autoencoder_foo"

    def test_resolve_model_name_regular(self):
        cfg = DictConfig({"model": {"name": "resnet"}, "dataset": {"params": {}}})
        name = test_utils._resolve_model_name(cfg)
        assert name == "resnet"

    def test_resolve_model_name_custom(self):
        cfg = DictConfig(
            {"model": {"name": "resnet"}, "dataset": {"params": {"path": "foo.zip"}}}
        )
        name = test_utils._resolve_model_name(cfg)
        assert name == "resnet_custom"
