"""
Helper functions for the train API.

This module contains internal helper functions used by the main train function
to handle configuration loading, model setup, and training pipeline preparation.

The module handles:
- Configuration loading and validation
- Logging setup for training sessions
- Pure ML training detection
- Registry modules and device setup
- Dataset, model, and loss function building
- Optimizer and scheduler configuration
- Trainer setup and initialization
- Training execution and metrics logging

These utilities ensure robust training pipeline setup with proper error handling
and configuration management for different model types and training scenarios.

Typical usage involves calling these helper functions from the main train
API to set up and execute training operations.
"""

import sys
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch

# Export for test compatibility
from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.train_utils import (
    _handle_fusion_training,
    _save_config_and_log_metrics,
    _setup_optimizer_config,
    _setup_trainer_params,
    load_config,
)

__all__ = [
    "_load_and_validate_config",
    "_setup_logging",
    "_check_pure_ml_training",
    "_get_modules_and_device",
    "_build_datasets_and_model",
    "_setup_optimizer_and_scheduler",
    "_setup_trainer",
    "_execute_training",
    "load_config",
    "OmegaConf",
]

# For tests expecting train_helpers.train_helpers
train_helpers = sys.modules[__name__]


def _load_and_validate_config(cfg: Union[str, DictConfig]) -> DictConfig:
    """
    Load and validate configuration for training.

    This function loads a configuration from a file path or validates
    an existing DictConfig object for training operations.

    Args:
        cfg: Configuration file path or DictConfig object

    Returns:
        Validated DictConfig object ready for training

    Note:
        This function ensures the configuration is properly loaded and
        validated before proceeding with training setup.
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
    Setup logging configuration for training.

    This function initializes or updates the logger for training operations,
    ensuring proper configuration logging and status reporting.

    Args:
        config: Configuration object containing logging settings
        resolved_model_name: Name of the model for logging identification
        logger: Optional existing logger instance

    Returns:
        Configured logger instance ready for training logging

    Raises:
        TypeError: If the configuration cannot be converted to a dictionary
    """
    from refrakt_core.api.utils.train_utils import setup_logger

    if logger is None:
        logger = setup_logger(config, resolved_model_name)

    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a dict after OmegaConf.to_container.")
    logger.log_config(cast(Dict[str, Any], config_dict))
    return logger


def _check_pure_ml_training(config: DictConfig) -> bool:
    """
    Check if this is a pure ML training session.

    This function detects whether the training session is for a pure ML model
    (non-deep learning) based on configuration parameters.

    Args:
        config: Configuration object to check for ML indicators

    Returns:
        True if this is a pure ML training session, False otherwise

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
    Get registry modules and device for training.

    This function initializes the registry modules (model, loss, trainer, wrapper)
    and determines the appropriate device for training operations.

    Returns:
        Tuple containing:
        - modules: Dictionary of registry functions for model building
        - device: torch.device instance for training operations
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


def _build_datasets_and_model(
    config: DictConfig,
    modules: Dict[str, Any],
    device: torch.device,
    logger: RefraktLogger,
) -> Tuple[Any, Any, Any, Any]:
    """
    Build datasets, loaders, and model for training.

    This function constructs the training and validation datasets, data loaders,
    model, and loss function using the provided configuration and modules.

    Args:
        config: Configuration object containing dataset, model, and loss settings
        modules: Registry dictionary containing available functions
        device: Target device for model placement
        logger: Logger instance for status messages

    Returns:
        Tuple containing:
        - train_loader: Training data loader
        - val_loader: Validation data loader
        - model: Configured model ready for training
        - loss_fn: Loss function for training
    """
    from refrakt_core.api.builders.loss_builder import build_loss
    from refrakt_core.api.builders.model_builder import build_model
    from refrakt_core.api.utils.train_utils import (
        build_datasets_and_loaders_with_resize,
    )

    train_dataset, val_dataset, train_loader, val_loader = (
        build_datasets_and_loaders_with_resize(config, logger)
    )
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

    return train_loader, val_loader, model, loss_fn


def _setup_optimizer_and_scheduler(
    config: DictConfig, model: torch.nn.Module
) -> Tuple[Any, Optional[Any]]:
    """
    Setup optimizer and scheduler for training.

    This function configures the optimizer and optional learning rate scheduler
    based on the training configuration.

    Args:
        config: Configuration object containing optimizer and scheduler settings
        model: Model whose parameters will be optimized

    Returns:
        Tuple containing:
        - optimizer: Configured optimizer instance
        - scheduler: Optional learning rate scheduler
    """
    opt_cls, optimizer_args = _setup_optimizer_config(config)
    optimizer = opt_cls(model.parameters(), **optimizer_args)

    scheduler: Optional[Any] = None
    if hasattr(config, "scheduler") and config.scheduler:
        from refrakt_core.api.builders.scheduler_builder import build_scheduler

        scheduler = build_scheduler(cast(OmegaConf, config), optimizer)

    return optimizer, scheduler


def _setup_trainer(
    config: DictConfig,
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    loss_fn: Any,
    optimizer: Any,
    scheduler: Any,
    device: str,
    modules: Dict[str, Any],
    artifact_dumper: Any,
    resolved_model_name: str,
    logger: RefraktLogger,
) -> Tuple[Any, int, str]:
    """
    Setup and initialize trainer for training.

    This function configures and initializes the appropriate trainer based on
    the model type and training configuration.

    Args:
        config: Configuration object containing trainer settings
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function for training
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        device: Target device string
        modules: Registry dictionary containing available functions
        artifact_dumper: Artifact dumper for saving outputs
        resolved_model_name: Name of the model for trainer configuration
        logger: Logger instance for status messages

    Returns:
        Tuple containing:
        - trainer: Configured trainer instance ready for training
        - num_epochs: Number of training epochs
        - final_device: Final device string for training
    """
    from refrakt_core.api.builders.trainer_builder import initialize_trainer

    trainer_cls, trainer_params, num_epochs, final_device = _setup_trainer_params(
        config, device, logger, artifact_dumper, resolved_model_name
    )

    trainer = initialize_trainer(
        cfg=cast(OmegaConf, config),
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=final_device,
        modules=modules,
        save_dir=trainer_params.get("save_dir"),
    )

    return trainer, num_epochs, final_device


def _execute_training(
    trainer: Any,
    num_epochs: int,
    config: DictConfig,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    final_device: str,
    artifact_dumper: Any,
    resolved_model_name: str,
    logger: RefraktLogger,
) -> Dict[str, Any]:
    """
    Execute the training process.

    This function runs the training loop, handles fusion head training if needed,
    and logs final metrics and configuration.

    Args:
        trainer: Configured trainer instance
        num_epochs: Number of training epochs
        config: Configuration object for logging
        model: Trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        final_device: Device string used for training
        artifact_dumper: Artifact dumper for saving outputs
        resolved_model_name: Name of the model for logging
        logger: Logger instance for status messages

    Returns:
        Dictionary containing final training metrics

    Note:
        This function handles both standard training and fusion head training,
        ensuring all components are properly trained and logged.
    """
    logger.info(f"\n🚀 Starting training for {num_epochs} epochs...")
    final_metrics = trainer.train(num_epochs)

    # Fusion Head Training
    _handle_fusion_training(
        config,
        model,
        train_loader,
        val_loader,
        final_device,
        artifact_dumper,
        trainer,
        logger,
    )

    # Save config and log final metrics
    _save_config_and_log_metrics(
        config, trainer, resolved_model_name, final_metrics, logger
    )

    # Ensure we return a Dict[str, Any] even if final_metrics is not a dict
    if isinstance(final_metrics, dict):
        return final_metrics
    else:
        return {"metrics": final_metrics}
