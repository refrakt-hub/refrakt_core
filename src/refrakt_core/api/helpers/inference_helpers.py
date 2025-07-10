"""
Helper functions for the inference API.

This module contains internal helper functions used by the main inference function
to handle configuration loading, model setup, and inference pipeline preparation.

The module handles:
- Configuration loading and validation
- Logging setup for inference sessions
- Pure ML inference detection
- Device setup and model loading
- Data loader configuration for inference

These utilities ensure robust inference pipeline setup with proper error handling
and configuration management for different model types and inference scenarios.

Typical usage involves calling these helper functions from the main inference
API to set up and execute inference operations.
"""

from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.inference_utils import setup_model_and_device

# Export for test compatibility
from refrakt_core.api.utils.train_utils import load_config, setup_logger

__all__ = [
    "_load_and_validate_config",
    "_setup_logging",
    "_check_pure_ml_inference",
    "_setup_device",
    "_load_model_and_setup",
    "_setup_data_loader",
    "load_config",
    "OmegaConf",
]


def _load_and_validate_config(cfg: Union[str, DictConfig]) -> DictConfig:
    """
    Load and validate configuration for inference.

    This function loads a configuration from a file path or validates
    an existing DictConfig object for inference operations.

    Args:
        cfg: Configuration file path or DictConfig object

    Returns:
        Validated DictConfig object ready for inference

    Note:
        This function ensures the configuration is properly loaded and
        validated before proceeding with inference setup.
    """
    return load_config(cfg)


def _setup_logging(
    config: DictConfig, resolved_model_name: str, logger: Optional[RefraktLogger]
) -> RefraktLogger:
    """
    Setup logging configuration for inference.

    This function initializes or updates the logger for inference operations,
    ensuring proper configuration logging and status reporting.

    Args:
        config: Configuration object containing logging settings
        resolved_model_name: Name of the model for logging identification
        logger: Optional existing logger instance

    Returns:
        Configured logger instance ready for inference logging
    """
    if logger is None:
        logger = setup_logger(config, resolved_model_name)

    config_dict = OmegaConf.to_container(config, resolve=True)
    logger.log_config(cast(Dict[str, Any], config_dict))
    return logger


def _check_pure_ml_inference(config: DictConfig) -> bool:
    """
    Check if this is a pure ML inference session.

    This function detects whether the inference session is for a pure ML model
    (non-deep learning) based on configuration parameters.

    Args:
        config: Configuration object to check for ML indicators

    Returns:
        True if this is a pure ML inference session, False otherwise

    Note:
        Pure ML models have different handling requirements compared to
        deep learning models, so this detection is important for proper setup.
    """
    return (
        getattr(config.model, "type", None) == "ml"
        or getattr(config.dataset, "name", None) == "tabular_ml"
    )


def _setup_device() -> torch.device:
    """
    Setup device for inference.

    This function determines the appropriate device (CPU or CUDA) for
    inference based on system availability.

    Returns:
        torch.device instance for inference operations
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model_and_setup(
    config: DictConfig, device: torch.device, model_path: str, logger: RefraktLogger
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model and setup for inference.

    This function loads the model from the specified path, sets it up on the
    target device, and prepares it for inference operations.

    Args:
        config: Configuration object containing model settings
        device: Target device for model placement
        model_path: Path to the model checkpoint file
        logger: Logger instance for status messages

    Returns:
        Tuple containing:
        - model: Loaded and configured model ready for inference
        - modules: Registry of available modules for the model
    """
    from refrakt_core.api.utils.test_utils import _load_model_checkpoint

    model, modules = setup_model_and_device(config, device)
    _load_model_checkpoint(model, model_path, device, logger)
    model.eval()

    return model, modules


def _setup_data_loader(config: DictConfig, data: Any, logger: RefraktLogger) -> Any:
    """
    Setup data loader for inference.

    This function configures a data loader for inference operations,
    including automatic image resizing if needed.

    Args:
        config: Configuration object containing data loader settings
        data: Input data for inference
        logger: Logger instance for status messages

    Returns:
        Configured data loader ready for inference
    """
    from refrakt_core.api.utils.train_utils import (
        setup_data_loader_for_inference_with_resize,
    )

    return setup_data_loader_for_inference_with_resize(config, data, logger)
