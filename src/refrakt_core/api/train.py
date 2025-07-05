"""
Train API for Refrakt.

This module provides the main train function for training models.
"""

import gc
import sys
import traceback
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.train_utils import (
    setup_artifact_dumper,
    _resolve_model_name_train,
    _handle_pure_ml_training,
)
from refrakt_core.api.helpers.train_helpers import (
    _load_and_validate_config,
    _setup_logging,
    _check_pure_ml_training,
    _get_modules_and_device,
    _build_datasets_and_model,
    _setup_optimizer_and_scheduler,
    _setup_trainer,
    _execute_training,
)


def train(
    cfg: Union[str, DictConfig], logger: Optional[RefraktLogger] = None
) -> Dict[str, Any]:
    """
    Orchestrate the training pipeline for Refrakt.

    Args:
        cfg (Union[str, DictConfig]): Path to config file or DictConfig object.
        logger (Optional[RefraktLogger]): Logger instance (optional).

    Returns:
        Dict[str, Any]: Training metrics and results.
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
            _handle_pure_ml_training(config, resolved_model_name, logger)
            return {"status": "completed", "type": "ml"}

        # Get modules and device
        modules, device = _get_modules_and_device()

        # Build datasets and model
        train_loader, val_loader, model, loss_fn = _build_datasets_and_model(config, modules, device, logger)

        # Setup optimizer and scheduler
        optimizer, scheduler = _setup_optimizer_and_scheduler(config, model)

        # Setup artifact dumper
        artifact_dumper = setup_artifact_dumper(config, resolved_model_name, logger)

        # Setup trainer
        trainer, num_epochs, final_device = _setup_trainer(
            config, model, train_loader, val_loader, loss_fn, optimizer, scheduler, 
            str(device), modules, artifact_dumper, resolved_model_name, logger
        )

        # Execute training
        return _execute_training(
            trainer, num_epochs, config, model, train_loader, val_loader, 
            final_device, artifact_dumper, resolved_model_name, logger
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
