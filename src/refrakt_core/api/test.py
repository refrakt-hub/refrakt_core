"""
Test API for Refrakt.

This module provides the main test function for evaluating
trained models using the Refrakt framework. It orchestrates the
complete testing pipeline including model loading, dataset preparation,
evaluation execution, and results reporting.

The module handles:
- Configuration loading and validation
- Model checkpoint loading and setup
- Dataset and dataloader preparation
- Model evaluation with various metrics
- Fusion model evaluation (if applicable)
- Results logging and reporting
- Error handling and cleanup
"""

import gc
import sys
import traceback
from typing import Optional, Union

import torch
from omegaconf import DictConfig

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.helpers.test_helpers import (
    _build_test_components,
    _check_pure_ml_testing,
    _evaluate_model,
    _get_modules_and_device,
    _load_and_validate_config,
    _setup_logging,
    _setup_trainer_for_testing,
)
from refrakt_core.api.utils.test_utils import (
    _handle_pure_ml_pipeline,
    _load_model_checkpoint,
    _resolve_model_name,
    _setup_fusion_evaluation,
)

__all__ = ["test", "_load_and_validate_config"]


def test(
    cfg: Union[str, DictConfig],
    model_path: Optional[str] = None,
    logger: Optional[RefraktLogger] = None,
) -> None:
    """
    Orchestrate the test pipeline for Refrakt.

    This function serves as the main entry point for evaluating trained models using the
    Refrakt framework. It handles the complete testing workflow including model loading,
    dataset preparation, evaluation execution, and results reporting.

    The function supports both deep learning and pure ML testing modes, automatically
    detecting the appropriate pipeline based on configuration. It also supports fusion
    model evaluation when applicable.

    Args:
        cfg: Path to configuration file (str) or DictConfig object containing all
            testing parameters including model, dataset, and evaluation settings
        model_path: Optional path to the model checkpoint file. If None, the path
            will be inferred from configuration
        logger: Optional RefraktLogger instance for logging. If None, a new logger
            will be created based on configuration

    Returns:
        None. Evaluation results are printed to console and logged to files.

    Raises:
        SystemExit: If testing fails due to configuration errors, model loading
            issues, or other critical failures. The function will log detailed error
            information before exiting.
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
        dataloader, model, loss_fn = _build_test_components(
            config, modules, device, logger
        )

        # Setup artifact dumper
        from refrakt_core.api.utils.train_utils import setup_artifact_dumper

        artifact_dumper = setup_artifact_dumper(config, resolved_model_name, logger)

        # Setup trainer for testing
        trainer = _setup_trainer_for_testing(
            config,
            model,
            dataloader,
            loss_fn,
            str(device),
            modules,
            artifact_dumper,
            resolved_model_name,
            logger,
        )

        # Load checkpoint
        _load_model_checkpoint(model, model_path, device, logger)

        # Setup fusion evaluation
        fusion_acc = _setup_fusion_evaluation(
            config, model, dataloader, device, artifact_dumper, logger
        )

        # Evaluate model performance
        eval_results = _evaluate_model(
            trainer, model, dataloader, device, fusion_acc, logger
        )

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


test = test
test._load_and_validate_config = _load_and_validate_config
