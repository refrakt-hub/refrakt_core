"""
Train API for Refrakt.

This module provides the main train function for training models
using the Refrakt framework. It orchestrates the complete training
pipeline including configuration management, model building,
dataset preparation, optimizer setup, and training execution.

The module handles:
- Configuration loading and validation
- Model and dataset building
- Optimizer and scheduler setup
- Training execution with logging
- Error handling and cleanup
- Memory management for GPU training
"""

import gc
import sys
import traceback
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.helpers.train_helpers import (
    _build_datasets_and_model,
    _check_pure_ml_training,
    _execute_training,
    _get_modules_and_device,
    _load_and_validate_config,
    _setup_logging,
    _setup_optimizer_and_scheduler,
    _setup_trainer,
)
from refrakt_core.api.utils.train_utils import (
    _handle_pure_ml_training,
    _resolve_model_name_train,
    setup_artifact_dumper,
)

__all__ = ["train", "_load_and_validate_config"]


def train(
    cfg: Union[str, DictConfig], logger: Optional[RefraktLogger] = None
) -> Dict[str, Any]:
    """
    Orchestrate the training pipeline for Refrakt.

    This function serves as the main entry point for training models using the Refrakt
    framework. It handles the complete training workflow including
    configuration management, model and dataset preparation,
    optimizer setup, and training execution.

    The function supports both deep learning and pure ML training modes, automatically
    detecting the appropriate pipeline based on configuration.

    Args:
        cfg: Path to configuration file (str) or DictConfig object containing all
            training parameters including model, dataset, optimizer, and training
            settings
        logger: Optional RefraktLogger instance for logging. If None, a new logger
            will be created based on configuration

    Returns:
        Dictionary containing training results, metrics, and status information.
        For pure ML training, returns {"status": "completed", "type": "ml"}.
        For deep learning training, returns comprehensive metrics and artifacts.

    Raises:
        SystemExit: If training fails due to configuration errors, model issues,
            or other critical failures. The function will log detailed error
            information before exiting.
    """
    try:
        torch.cuda.empty_cache()
        # Load and validate configuration
        config = _load_and_validate_config(cfg)
        resolved_model_name = _resolve_model_name_train(config)

        # Setup logging
        logger = _setup_logging(config, resolved_model_name, logger)

        # Check for pure ML training
        if _check_pure_ml_training(config):
            return _handle_pure_ml_training(config, resolved_model_name, logger)

        # Get modules and device
        modules, device = _get_modules_and_device()

        # Build datasets and model
        train_loader, val_loader, model, loss_fn = _build_datasets_and_model(
            config, modules, device, logger
        )

        # Setup optimizer and scheduler
        optimizer, scheduler = _setup_optimizer_and_scheduler(config, model)

        # Setup artifact dumper
        artifact_dumper = setup_artifact_dumper(config, resolved_model_name, logger)

        # Setup trainer
        trainer, num_epochs, final_device = _setup_trainer(
            config,
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            scheduler,
            str(device),
            modules,
            artifact_dumper,
            resolved_model_name,
            logger,
        )

        # Execute training
        return _execute_training(
            trainer,
            num_epochs,
            config,
            model,
            train_loader,
            val_loader,
            final_device,
            artifact_dumper,
            resolved_model_name,
            logger,
        )

    except Exception as e:
        if not (logger and hasattr(logger, "error")):
            logger = None
        if logger:
            logger.error(f"\n❌ Training failed: {str(e)}")
            logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        gc.collect()
        torch.cuda.empty_cache()
