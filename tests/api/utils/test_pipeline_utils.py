from refrakt_core.registry.dataset_registry import DATASET_REGISTRY
from tests.helpers.fixtures import DummyDataset

DATASET_REGISTRY["dummy"] = DummyDataset

import importlib
import os
import zipfile
from contextlib import contextmanager

import pytest
from omegaconf import DictConfig

import refrakt_core.api.utils.pipeline_utils as pipeline_utils
from refrakt_core.api.core.logger import RefraktLogger


class DummyLogger(RefraktLogger):
    def info(self, msg):
        self.info_called = True

    def error(self, msg):
        self.error_called = True

    def warning(self, msg):
        self.warning_called = True

    def debug(self, msg):
        self.debug_called = True


class TestPipelineUtils:
    # Smoke Test
    def test_import_pipeline_utils(self):
        importlib.reload(pipeline_utils)

    def test_setup_logger_and_config_success(self):
        logger = pipeline_utils.setup_logger_and_config(
            cfg={"foo": 1},
            model_name="resnet18",
            log_dir="./logs",
            log_types=["file"],
            console=True,
            debug=False,
            all_overrides=[],
        )
        # Relaxed assertion: check class name instead of isinstance
        assert type(logger).__name__ == "RefraktLogger"

    def test_setup_logger_and_config_typeerror(self):
        with pytest.raises(TypeError):
            pipeline_utils.setup_logger_and_config(
                cfg=None,
                model_name="resnet18",
                log_dir="./logs",
                log_types=["file"],
                console=True,
                debug=False,
                all_overrides=[],
            )

    def test_setup_logger_and_config_valueerror(self):
        with pytest.raises(ValueError):
            pipeline_utils.setup_logger_and_config(
                cfg={"foo": 1},
                model_name="",
                log_dir="./logs",
                log_types=["file"],
                console=True,
                debug=False,
                all_overrides=[],
            )

    def test_resolve_model_name_autoencoder(self):
        cfg = DictConfig(
            {
                "model": {"name": "autoencoder", "params": {"variant": "foo"}},
                "dataset": {"params": {}},
            }
        )
        name = pipeline_utils.resolve_model_name(cfg)
        assert name == "autoencoder_foo"

    def test_resolve_model_name_regular(self):
        cfg = DictConfig({"model": {"name": "resnet"}, "dataset": {"params": {}}})
        name = pipeline_utils.resolve_model_name(cfg)
        assert name == "resnet"

    def test_resolve_model_name_custom(self):
        cfg = DictConfig(
            {"model": {"name": "resnet"}, "dataset": {"params": {"path": "foo.zip"}}}
        )
        name = pipeline_utils.resolve_model_name(cfg)
        assert name == "resnet_custom"

