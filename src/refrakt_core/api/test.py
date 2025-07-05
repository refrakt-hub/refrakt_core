"""
Test API for Refrakt.

This module provides the main test function for evaluating trained models.
"""

import gc
import sys
import traceback
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.test_utils import (
    _resolve_model_name,
    _handle_pure_ml_pipeline,
    _setup_fusion_evaluation,
    _load_model_checkpoint,
)
from refrakt_core.api.helpers.test_helpers import (
    _load_and_validate_config,
    _setup_logging,
    _check_pure_ml_testing,
    _get_modules_and_device,
    _build_test_components,
    _setup_trainer_for_testing,
    _evaluate_model,
)


def test(
    cfg: Union[str, DictConfig], model_path: Optional[str] = None, logger: Optional[RefraktLogger] = None
) -> None:
    """
    Orchestrate the test pipeline for Refrakt.

    Args:
        cfg (Union[str, DictConfig]): Path to config file or DictConfig object.
        model_path (Optional[str]): Path to the model checkpoint.
        logger (Optional[RefraktLogger]): Logger instance (optional).
    """
    try:
        # Load and validate configuration
        config = _load_and_validate_config(cfg)
        resolved_model_name = _resolve_model_name(config)

        # Setup logging
        logger = _setup_logging(config, resolved_model_name, logger)

        # Check for pure ML testing
        if _check_pure_ml_testing(config):
            _handle_pure_ml_pipeline(config, resolved_model_name, logger)
            return

        # Get modules and device
        modules, device = _get_modules_and_device()

        # Build test components
        dataloader, model, loss_fn = _build_test_components(config, modules, device, logger)

        # Setup artifact dumper
        from refrakt_core.api.utils.train_utils import setup_artifact_dumper
        artifact_dumper = setup_artifact_dumper(config, resolved_model_name, logger)

        # Setup trainer for testing
        trainer = _setup_trainer_for_testing(
            config, model, dataloader, loss_fn, str(device), modules, 
            artifact_dumper, resolved_model_name, logger
        )

        # Load checkpoint
        _load_model_checkpoint(model, model_path, device, logger)

        # Setup fusion evaluation
        fusion_acc = _setup_fusion_evaluation(config, model, dataloader, device, artifact_dumper, logger)

        # Evaluate model performance
        eval_results = _evaluate_model(trainer, model, dataloader, device, fusion_acc, logger)

        logger.info("\n✅ Testing completed successfully!")
        print("\nEvaluation Results:", eval_results)

    except Exception as e:
        if not (logger and hasattr(logger, "error")):
            logger = None
        if logger:
            logger.error(f"\n❌ Testing failed: {str(e)}")
            logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        gc.collect()
        torch.cuda.empty_cache()
