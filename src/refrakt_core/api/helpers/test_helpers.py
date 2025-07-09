"""
Helper functions for the test API.

This module contains internal helper functions used by the main test function
to handle configuration loading, model setup, and testing pipeline preparation.

The module handles:
- Configuration loading and validation
- Logging setup for testing sessions
- Pure ML testing detection
- Registry modules and device setup
- Test components building (dataloader, model, loss)
- Trainer setup for testing
- Model evaluation and performance assessment

These utilities ensure robust testing pipeline setup with proper error handling
and configuration management for different model types and testing scenarios.

Typical usage involves calling these helper functions from the main test
API to set up and execute testing operations.
"""

from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.test_utils import (
    _build_test_loader_with_resize,
    _run_manual_evaluation,
)
from refrakt_core.api.utils.train_utils import load_config, setup_logger


def _load_and_validate_config(cfg: Union[str, DictConfig]) -> DictConfig:
    """
    Load and validate configuration for testing.

    This function loads a configuration from a file path or validates
    an existing DictConfig object for testing operations.

    Args:
        cfg: Configuration file path or DictConfig object

    Returns:
        Validated DictConfig object ready for testing

    Note:
        This function ensures the configuration is properly loaded and
        validated before proceeding with testing setup.
    """
    if isinstance(cfg, str):
        config = load_config(cfg)
    else:
        config = cfg
    return config


def _setup_logging(
    config: DictConfig, resolved_model_name: str, logger: Optional[RefraktLogger]
) -> RefraktLogger:
    """
    Setup logging configuration for testing.

    This function initializes or updates the logger for testing operations,
    ensuring proper configuration logging and status reporting.

    Args:
        config: Configuration object containing logging settings
        resolved_model_name: Name of the model for logging identification
        logger: Optional existing logger instance

    Returns:
        Configured logger instance ready for testing

    Raises:
        TypeError: If the configuration cannot be converted to a dictionary
    """
    if logger is None:
        logger = setup_logger(config, resolved_model_name)

    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a dict after OmegaConf.to_container.")
    logger.log_config(cast(Dict[str, Any], config_dict))
    return logger


def _check_pure_ml_testing(config: DictConfig) -> bool:
    """
    Check if this is a pure ML testing session.

    This function detects whether the testing session is for a pure ML model
    (non-deep learning) based on configuration parameters.

    Args:
        config: Configuration object to check for ML indicators

    Returns:
        True if this is a pure ML testing session, False otherwise

    Note:
        Pure ML models have different handling requirements compared to
        deep learning models, so this detection is important for proper setup.
    """
    return (
        getattr(config.model, "type", None) == "ml"
        or getattr(config.dataset, "name", None) == "tabular_ml"
    )


def _get_modules_and_device() -> Tuple[Dict[str, Any], torch.device]:
    """
    Get registry modules and device for testing.

    This function initializes the registry modules (model, loss, trainer, wrapper)
    and determines the appropriate device for testing operations.

    Returns:
        Tuple containing:
        - modules: Dictionary of registry functions for model building
        - device: torch.device instance for testing operations
    """
    from refrakt_core.registry.loss_registry import get_loss
    from refrakt_core.registry.model_registry import get_model
    from refrakt_core.registry.trainer_registry import get_trainer
    from refrakt_core.registry.wrapper_registry import get_wrapper

    modules: Dict[str, Any] = {
        "get_model": get_model,
        "get_loss": get_loss,
        "get_trainer": get_trainer,
        "get_wrapper": get_wrapper,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return modules, device


def _build_test_components(
    config: DictConfig,
    modules: Dict[str, Any],
    device: torch.device,
    logger: RefraktLogger,
) -> Tuple[Any, Any, Any]:
    """
    Build test components (dataloader, model, loss).

    This function constructs the test data loader, model, and loss function
    using the provided configuration and modules.

    Args:
        config: Configuration object containing dataset, model, and loss settings
        modules: Registry dictionary containing available functions
        device: Target device for model placement
        logger: Logger instance for status messages

    Returns:
        Tuple containing:
        - dataloader: Test data loader
        - model: Configured model ready for testing
        - loss_fn: Loss function for evaluation
    """
    from refrakt_core.api.builders.loss_builder import build_loss
    from refrakt_core.api.builders.model_builder import build_model

    dataloader = _build_test_loader_with_resize(config, logger)
    model_cls = modules["get_model"](config.model.name)

    model = build_model(
        cast(OmegaConf, config),
        modules={
            "get_model": modules["get_model"],
            "get_wrapper": modules["get_wrapper"],
            "model": model_cls,
        },
        device=str(device),
    )

    loss_fn = build_loss(cast(OmegaConf, config), modules=modules, device=str(device))

    return dataloader, model, loss_fn


def _setup_trainer_for_testing(
    config: DictConfig,
    model: torch.nn.Module,
    dataloader: Any,
    loss_fn: Any,
    device: str,
    modules: Dict[str, Any],
    artifact_dumper: Any,
    resolved_model_name: str,
    logger: RefraktLogger,
) -> Any:
    """
    Setup trainer for testing.

    This function configures a trainer specifically for testing operations,
    setting up the model, data loader, and loss function for evaluation.

    Args:
        config: Configuration object containing trainer settings
        model: Model to test
        dataloader: Test data loader
        loss_fn: Loss function for evaluation
        device: Target device string
        modules: Registry dictionary containing available functions
        artifact_dumper: Artifact dumper for saving outputs
        resolved_model_name: Name of the model for trainer configuration
        logger: Logger instance for status messages

    Returns:
        Configured trainer instance ready for testing
    """
    from refrakt_core.api.builders.trainer_builder import initialize_trainer

    trainer = initialize_trainer(
        cfg=cast(OmegaConf, config),
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        loss_fn=loss_fn,
        optimizer=None,
        scheduler=None,
        device=device,
        modules=modules,
        save_dir=None,
    )
    trainer.model_name = resolved_model_name
    trainer.logger = logger
    trainer.artifact_dumper = artifact_dumper

    return trainer


def _evaluate_model(
    trainer: Any,
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    fusion_acc: Optional[float],
    logger: RefraktLogger,
) -> Dict[str, Any]:
    """
    Evaluate model performance.

    This function evaluates the model's performance using either the trainer's
    evaluate method or manual evaluation, handling different model types and
    evaluation scenarios.

    Args:
        trainer: Trainer instance that may have an evaluate method
        model: Model to evaluate
        dataloader: Test data loader
        device: Target device for evaluation
        fusion_acc: Optional fusion accuracy for fusion models
        logger: Logger instance for status messages

    Returns:
        Dictionary containing evaluation results and metrics

    Note:
        This function handles both standard model evaluation and fusion model
        evaluation, providing appropriate metrics for each case.
    """
    model.eval()
    eval_results = {}

    # Use trainer's evaluate method if available
    if hasattr(trainer, "evaluate"):
        try:
            if fusion_acc is not None:
                # For fusion models, we already evaluated above
                eval_results["fusion_accuracy"] = fusion_acc
            else:
                # For regular models, use trainer's evaluate method
                accuracy = trainer.evaluate()
                eval_results["accuracy"] = accuracy
                logger.info(f"Model accuracy: {accuracy:.4f}")
        except Exception as e:
            logger.warning(f"Could not use trainer's evaluate method: {e}")
            eval_results = _run_manual_evaluation(model, dataloader, device, logger)
    else:
        # Manual evaluation if trainer doesn't have evaluate method
        eval_results = _run_manual_evaluation(model, dataloader, device, logger)

    return eval_results
