import logging
from unittest.mock import MagicMock, patch

import pytest

from refrakt_core.api.core.logger import RefraktLogger


def test_logger_smoke():
    logger = RefraktLogger(
        "test_model", log_types=["tensorboard", "wandb"], console=True, debug=True
    )
    assert isinstance(logger, RefraktLogger)
    logger.close()


def test_logger_sanity():
    logger = RefraktLogger("test_model", log_types=[], console=False, debug=False)
    logger.info("info message")
    logger.warning("warn message")
    logger.error("error message")
    logger.debug("debug message")
    logger.close()


@patch("torch.utils.tensorboard.SummaryWriter", autospec=True)
def test_tensorboard_init(mock_summary_writer):
    logger = RefraktLogger("tb_model", log_types=["tensorboard"])
    assert logger.tb_writer is not None
    logger.close()


@patch("refrakt_core.api.core.logger.wandb", create=True)
def test_wandb_init(mock_wandb):
    mock_wandb.init.return_value = MagicMock()
    logger = RefraktLogger("wandb_model", log_types=["wandb"])
    assert logger.wandb_run is not None
    logger.close()


def test_log_metrics_and_config(monkeypatch):
    logger = RefraktLogger("test_model")
    logger.tb_writer = MagicMock()
    logger.wandb_run = MagicMock()
    logger.log_metrics({"acc": 0.9}, step=1)
    logger.log_config({"param": 1})
    logger.close()


def test_log_images_and_inference(monkeypatch):
    import numpy as np

    logger = RefraktLogger("test_model")
    logger.tb_writer = MagicMock()
    logger.wandb_run = MagicMock()
    images = np.random.rand(2, 3, 32, 32)
    logger.log_images("test", images, step=0)
    logger.log_inference_results(images, images, images, step=0)
    logger.close()


def test_logger_close_safe():
    logger = RefraktLogger("test_model")
    # Should not raise even if tb_writer and wandb_run are None
    logger.close()
