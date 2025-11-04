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

import json
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch

# Export for test compatibility
from omegaconf import DictConfig, OmegaConf

from refrakt_cli.helpers.shared_core import extract_comprehensive_metadata
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.hooks_orchestrator import (  # type: ignore
    instantiate_explainability_hooks,
    instantiate_visualization_hooks,
)
from refrakt_core.api.utils.pipeline_utils import parse_runtime_hooks
from refrakt_core.api.utils.train_utils import (
    _handle_fusion_training,
    _save_config_and_log_metrics,
    _setup_optimizer_config,
    _setup_trainer_params,
    build_optimizer_and_scheduler,
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
    "_save_test_summary_metrics",
    "_save_inference_summary_metrics",
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
        logger=logger,
    )

    # for name, module in model.named_modules():
    #     print(f" {name}: {module}")

    loss_fn = build_loss(
        cast(OmegaConf, config), modules=modules, device=str(device), logger=logger
    )

    return train_loader, val_loader, model, loss_fn


def _setup_optimizer_and_scheduler(
    config: DictConfig, model: torch.nn.Module, logger: Optional[RefraktLogger] = None
) -> Tuple[Any, Optional[Any]]:
    """
    Setup optimizer and scheduler for training.

    This function configures the optimizer and optional learning rate scheduler
    based on the training configuration. It supports both standard single optimizers
    and GAN-style nested optimizer structures.

    Args:
        config: Configuration object containing optimizer and scheduler settings
        model: Model whose parameters will be optimized
        logger: Optional logger instance for debug output

    Returns:
        Tuple containing:
        - optimizer: Configured optimizer instance or dictionary for GAN models
        - scheduler: Optional learning rate scheduler
    """
    # Use the proper optimizer builder that supports GAN nested structures
    return build_optimizer_and_scheduler(config, model, logger)


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
    experiment_id: Optional[str] = None,
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
        experiment_id: Optional experiment ID for consistent directory naming

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

    # --- Inject hook instantiation here ---
    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        config_dict = {}
    viz_hooks, xai_hooks, explain_flag = parse_runtime_hooks(
        cast(Dict[str, Any], config_dict)
    )
    # Example: pass class_names if available (for supervised)
    class_names = None
    if hasattr(config, "dataset") and hasattr(config.dataset, "params"):
        class_names = getattr(config.dataset.params, "class_names", None)
    # Fallback: try to infer class_names from num_classes
    if class_names is None:
        num_classes = None
        if hasattr(config, "model") and hasattr(config.model, "params"):
            num_classes = getattr(config.model.params, "num_classes", None)
        if num_classes is not None:
            class_names = [str(i) for i in range(num_classes)]
    viz_components = instantiate_visualization_hooks(
        viz_hooks, extra_args={"class_names": class_names} if class_names else {}
    )
    xai_components = instantiate_explainability_hooks(xai_hooks)
    # --- End hook injection ---

    # Remove save_dir from trainer_params to avoid duplicate keyword argument
    save_dir = trainer_params.pop("save_dir", None)
    # Remove keys not accepted by initialize_trainer
    for k in [
        "logger",
        "artifact_dumper",
        "visualization_hooks",
        "explainability_hooks",
        "model_name",
    ]:
        trainer_params.pop(k, None)

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
        save_dir=save_dir,
        experiment_id=experiment_id,
        logger=logger,
        **trainer_params,
    )
    # Set model_name on the trainer instance for correct checkpoint naming
    setattr(trainer, "model_name", resolved_model_name)
    # Set experiment directory if artifact_dumper provides it
    if hasattr(artifact_dumper, "experiment_dir"):
        setattr(trainer, "experiment_dir", artifact_dumper.experiment_dir)
    # Set visualization and explainability hooks as attributes if present
    if viz_components:
        setattr(trainer, "visualization_hooks", viz_components)
    if xai_components:
        setattr(trainer, "explainability_hooks", xai_components)

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
    experiment_id: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the training process and save results in the new directory structure.
    """
    # Execute training
    training_results = trainer.train(num_epochs)

    if logger:
        logger.info(f"Training results: {training_results}")

    # Handle fusion training if configured and merge results
    fusion_results = _handle_fusion_training(
        config,
        model,
        train_loader,
        val_loader,
        final_device,
        artifact_dumper,
        trainer,
        logger,
    )

    # Merge fusion results with training results
    if fusion_results:
        if isinstance(training_results, dict) and isinstance(fusion_results, dict):
            training_results.update(fusion_results)
        elif logger:
            logger.info(f"Fusion training results: {fusion_results}")

    # Save summary metrics in experiment directory
    # Use checkpoint directory if experiment_dir is not available
    experiment_dir = None
    if hasattr(trainer, "experiment_dir") and trainer.experiment_dir:
        experiment_dir = trainer.experiment_dir
    elif hasattr(trainer, "save_dir") and trainer.save_dir and experiment_id:
        # The save_dir is typically ./checkpoints/autoencoder_vae_{experiment_id}/weights
        # So we need to go up one level to get the experiment directory
        experiment_dir = os.path.dirname(
            trainer.save_dir
        )  # This should be ./checkpoints/autoencoder_vae_{experiment_id}

    if experiment_dir:
        summary_metrics_path = os.path.join(
            experiment_dir, "explanations", "summary_metrics.json"
        )

        # Create explanations directory if it doesn't exist
        os.makedirs(os.path.dirname(summary_metrics_path), exist_ok=True)

        # Extract comprehensive metadata
        metadata = extract_comprehensive_metadata(
            [config_path] if config_path else [],
            os.getcwd(),
            experiment_dir,
            logger,
            training_results,
        )

        # --- Robust summary_metrics merging logic ---
        if os.path.exists(summary_metrics_path):
            try:
                with open(summary_metrics_path, "r") as f:
                    existing_metrics = json.load(f)
                # Merge experiment_info, model_info, dataset_info, trainer_info, optimizer_info, run_metadata
                for key in [
                    "experiment_info",
                    "model_info",
                    "dataset_info",
                    "trainer_info",
                    "optimizer_info",
                    "run_metadata",
                ]:
                    if key in existing_metrics and key in metadata:
                        if not metadata[key]:
                            metadata[key] = existing_metrics[key]
                # Merge performance_metrics, preferring most complete (non-'N/A') values
                merged_perf = existing_metrics.get("performance_metrics", {}).copy()
                for k, v in metadata.get("performance_metrics", {}).items():
                    if v not in [None, "", "N/A"]:
                        merged_perf[k] = v
                    elif k in merged_perf and merged_perf[k] not in [None, "", "N/A"]:
                        continue  # keep existing good value
                    else:
                        merged_perf[k] = v  # fallback
                metadata["performance_metrics"] = merged_perf
            except Exception as e:
                if logger:
                    logger.warning(
                        f"Could not merge with existing summary_metrics.json: {e}"
                    )
        # Save to experiment directory
        with open(summary_metrics_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if logger:
            logger.info(f"Summary metrics saved to {summary_metrics_path}")

    return training_results


def _save_test_summary_metrics(
    trainer, eval_results, resolved_model_name, logger, experiment_id, config_files=None
):
    """Save summary_metrics.json for test phase, using config_files for metadata."""
    experiment_dir = None
    if hasattr(trainer, "experiment_dir") and trainer.experiment_dir:
        experiment_dir = trainer.experiment_dir
    elif hasattr(trainer, "save_dir") and trainer.save_dir and experiment_id:
        experiment_dir = os.path.dirname(trainer.save_dir)
    if experiment_dir:
        summary_metrics_path = os.path.join(
            experiment_dir, "explanations", "summary_metrics.json"
        )
        os.makedirs(os.path.dirname(summary_metrics_path), exist_ok=True)
        base_dir = os.getcwd()
        checkpoints_dir = experiment_dir
        metadata = extract_comprehensive_metadata(
            config_files or [], base_dir, checkpoints_dir, logger, eval_results
        )
        with open(summary_metrics_path, "w") as f:
            json.dump(metadata, f, indent=2)
        if logger:
            logger.info(f"Summary metrics saved to {summary_metrics_path}")


def _save_inference_summary_metrics(
    model, results, resolved_model_name, logger, experiment_id, config_files=None
):
    """Save summary_metrics.json for inference phase, using config_files for metadata."""
    experiment_dir = None
    if hasattr(model, "experiment_dir") and model.experiment_dir:
        experiment_dir = model.experiment_dir
    else:
        experiment_dir = os.path.join(
            "./checkpoints", f"{resolved_model_name}_{experiment_id}"
        )
    if experiment_dir:
        summary_metrics_path = os.path.join(
            experiment_dir, "explanations", "summary_metrics.json"
        )
        os.makedirs(os.path.dirname(summary_metrics_path), exist_ok=True)
        base_dir = os.getcwd()
        checkpoints_dir = experiment_dir
        metrics = results[-1] if isinstance(results, list) and results else {}
        # --- Patch: Convert ModelOutput to dict if needed ---
        from refrakt_core.schema.model_output import ModelOutput

        if isinstance(metrics, ModelOutput):
            metrics = metrics.summary()
        metadata = extract_comprehensive_metadata(
            config_files or [], base_dir, checkpoints_dir, logger, metrics
        )
        with open(summary_metrics_path, "w") as f:
            json.dump(metadata, f, indent=2)
        if logger:
            logger.info(f"Summary metrics saved to {summary_metrics_path}")
