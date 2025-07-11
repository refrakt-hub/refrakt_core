"""
Pipeline utilities for Refrakt.

This module provides comprehensive utility functions for pipeline execution,
extracted from the main __main__.py to reduce complexity and improve maintainability.

The module handles:
- Logger setup and configuration management
- Training pipeline execution and coordination
- Testing pipeline execution and validation
- Inference pipeline execution and prediction
- Full pipeline orchestration (train → test → inference)
- Model name resolution for pipeline operations
- Configuration override application and validation

These utilities ensure robust pipeline execution with proper error handling,
comprehensive logging, and coordinated execution of different pipeline phases.

Typical usage involves calling these utility functions to set up and execute
complete pipelines with automatic phase coordination and logging.
"""

import os
from typing import Any, cast

from omegaconf import DictConfig

import refrakt_core.api as refrakt_api
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.global_logging import set_global_logger


def setup_logger_and_config(
    cfg: Any,
    model_name: str,
    log_dir: str,
    log_types: list[str],
    console: bool,
    debug: bool,
    all_overrides: list[str],
) -> RefraktLogger:
    """
    Setup logger and apply configuration overrides.

    Args:
        cfg: Configuration object
        model_name: Name of the model
        log_dir: Directory for logs
        log_types: Types of logging to enable
        console: Whether to log to console
        debug: Whether to enable debug logging
        all_overrides: List of configuration overrides

    Returns:
        Configured logger instance
    """
    # Type and value checks
    if not isinstance(cfg, (dict, DictConfig)):
        raise TypeError("cfg must be a dict or DictConfig")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(log_dir, str) or not log_dir:
        raise ValueError("log_dir must be a non-empty string")
    logger = RefraktLogger(
        model_name=model_name,
        log_dir=log_dir,
        log_types=log_types,
        console=console,
        debug=debug,
    )

    logger.info(f"Logging initialized. Log file: {logger.log_file}")
    if all_overrides:
        logger.info(f"Applied overrides: {all_overrides}")
    set_global_logger(logger.logger)

    return logger


def execute_training_pipeline(cfg: Any, model_path: str, logger: RefraktLogger) -> None:
    """
    Execute the training pipeline.

    Args:
        cfg: Configuration object
        model_path: Path to model checkpoint
        logger: Logger instance
    """
    logger.info(f"Starting training with config: {cfg}")
    refrakt_api.train.train(cast("str | DictConfig", cfg), logger=logger)


def execute_testing_pipeline(cfg: Any, model_path: str, logger: RefraktLogger) -> None:
    """
    Execute the testing pipeline.

    Args:
        cfg: Configuration object
        model_path: Path to model checkpoint
        logger: Logger instance
    """
    logger.info(f"Starting testing with config: {cfg}")
    refrakt_api.test.test(
        cast("str | DictConfig", cfg), model_path=model_path, logger=logger
    )


def execute_inference_pipeline(
    cfg: Any, model_path: str, logger: RefraktLogger
) -> None:
    """
    Execute the inference pipeline.

    Args:
        cfg: Configuration object
        model_path: Path to model checkpoint
        logger: Logger instance
    """
    logger.info(f"Starting inference with config: {cfg}")
    refrakt_api.inference.inference(
        cast("str | DictConfig", cfg), model_path=model_path, logger=logger
    )


def execute_full_pipeline(cfg: Any, logger: RefraktLogger) -> None:
    """
    Execute the full pipeline (train → test → inference).

    Args:
        cfg: Configuration object
        logger: Logger instance
    """
    logger.info("🔁 Starting full pipeline (train → test → inference)")
    save_dir = cfg.trainer.params.save_dir

    # Resolve model name consistently with train/test phases
    if cfg.model.name == "autoencoder":
        variant = cfg.model.params.get("variant", "simple")
        resolved_model_name = f"autoencoder_{variant}"
    else:
        resolved_model_name = cfg.model.name

    # Check if using custom dataset and append _custom suffix
    dataset_params = (
        cfg.dataset.params
        if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "params")
        else {}
    )
    dataset_path = dataset_params.get("path", "") or dataset_params.get("zip_path", "")
    if dataset_path and str(dataset_path).endswith(".zip"):
        resolved_model_name = f"{resolved_model_name}_custom"

    model_path = os.path.join(save_dir, f"{resolved_model_name}.pth")

    logger.info("🚀 Training phase started")
    refrakt_api.train.train(cast("str | DictConfig", cfg), logger=logger)

    logger.info("🧪 Testing phase started")
    refrakt_api.test.test(
        cast("str | DictConfig", cfg), model_path=model_path, logger=logger
    )

    logger.info("🔮 Inference phase started")
    refrakt_api.inference.inference(
        cast("str | DictConfig", cfg), model_path=model_path, logger=logger
    )


def resolve_model_name(cfg: Any) -> str:
    """
    Resolve model name for pipeline execution.

    Args:
        cfg: Configuration object

    Returns:
        Resolved model name
    """
    if cfg.model.name == "autoencoder":
        variant = cfg.model.params.get("variant", "simple")
        resolved_model_name = f"autoencoder_{variant}"
    else:
        resolved_model_name = cfg.model.name

    # Check if using custom dataset and append _custom suffix
    dataset_params = (
        cfg.dataset.params
        if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "params")
        else {}
    )
    dataset_path = dataset_params.get("path", "") or dataset_params.get("zip_path", "")
    if dataset_path and str(dataset_path).endswith(".zip"):
        resolved_model_name = f"{resolved_model_name}_custom"

    return resolved_model_name
