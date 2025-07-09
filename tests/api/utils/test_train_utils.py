import importlib

import pytest
import torch
from omegaconf import DictConfig

import src.refrakt_core.api.utils.train_utils as train_utils
from src.refrakt_core.api.core.logger import RefraktLogger


class DummyLogger(RefraktLogger):
    def info(self, msg):
        self.info_called = True

    def warning(self, msg):
        self.warning_called = True


class DummyDataset:
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return torch.ones(3, 32, 32)


class TestTrainUtils:
    # Smoke Test
    def test_import_train_utils(self):
        importlib.reload(train_utils)

    # Sanity Tests
    def test_load_config_dictconfig(self):
        cfg = DictConfig({"foo": 1})
        out = train_utils.load_config(cfg)
        assert out == cfg

    def test_setup_logger(self):
        cfg = DictConfig(
            {
                "runtime": {
                    "log_type": ["file"],
                    "log_dir": "./logs",
                    "console": True,
                    "debug": False,
                }
            }
        )
        logger = train_utils.setup_logger(cfg, "resnet")
        assert hasattr(logger, "info") and hasattr(logger, "warning")

    def test_analyze_and_resize_dataset_images_empty(self):
        ds = DummyDataset()
        logger = DummyLogger("resnet", "./logs", ["file"], True, False)
        needs_resize, modified = train_utils.analyze_and_resize_dataset_images(
            ds, logger
        )
        assert needs_resize is False
        assert modified is ds
