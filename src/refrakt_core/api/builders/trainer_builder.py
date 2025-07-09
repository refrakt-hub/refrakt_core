"""
Trainer builder for Refrakt.

This module provides utilities to construct and initialize trainers from \
    configuration dictionaries. It supports standard, GAN, and fusion trainers, \
    and ensures robust type checking and flexible parameter handling.

The module handles:
- Trainer configuration validation and parsing
- Standard trainer setup for supervised learning, autoencoders, and MSN
- GAN trainer setup with generator/discriminator components
- Fusion trainer setup for ensemble models
- Artifact dumping and checkpoint management
- Device placement and trainer optimization
- Fallback trainer configuration for unsupported types

Typical usage involves passing a configuration (OmegaConf), model, dataloaders, \
    loss, optimizer, scheduler, and other components to build a trainer for \
    training or evaluation.
"""

from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from refrakt_core.registry.trainer_registry import register_trainer

from .utils.trainer_utils import (
    setup_fallback_trainer,
    setup_fusion_trainer,
    setup_gan_trainer,
    setup_standard_trainer,
    validate_trainer_config,
)


@register_trainer("ml")
class DummyMLTrainer:
    """Dummy trainer for ML-only pipelines that don't require deep learning training."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize dummy trainer with any arguments."""
        pass  # Dummy trainer does nothing

    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Dummy training method that does nothing."""
        pass  # Dummy training method

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Dummy evaluation method that does nothing."""
        pass  # Dummy evaluation method


def initialize_trainer(
    cfg: OmegaConf,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    loss_fn: Any,
    optimizer: Any,
    scheduler: Any,
    device: str,
    modules: Dict[str, Any],
    save_dir: Optional[str],
) -> Any:
    """
    Initialize a trainer based on configuration and provided components.

    This function supports standard, GAN, and fusion trainers, and ensures all \
        parameters are type-checked and compatible with the training pipeline. \
        It handles special cases for GAN and fusion trainers, including artifact \
        dumping and fusion head construction.

    The function automatically detects the trainer type and sets up the appropriate
    configuration:
    - Standard: Supervised learning, autoencoders, MSN
    - GAN: Generator/discriminator training with specialized setup
    - Fusion: Ensemble model training with fusion components
    - Fallback: Generic trainer for unsupported types

    Args:
        cfg: Configuration object (OmegaConf) specifying the trainer type,
             parameters, and optional settings
        model: The model to be trained. Can be a single model or model components
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        loss_fn: Loss function or dictionary of loss functions for \
            multi-component setups
        optimizer: Optimizer or dictionary of optimizers for \
            multi-component setups
        scheduler: Learning rate scheduler or dictionary of schedulers
        device: Target device string (e.g., "cuda", "cpu") for trainer placement
        modules: Registry dictionary containing available trainer, artifact, and \
            utility functions
        save_dir: Optional directory path for saving checkpoints and artifacts

    Returns:
        The instantiated trainer object, ready for training or evaluation.
        The exact type depends on the trainer configuration.

    Raises:
        TypeError: If the configuration or its fields are not of the expected type
        ValueError: If required trainer or fusion components are missing or not found
                   in the registry
    """
    print("Initializing trainer...")
    if OmegaConf.is_config(cfg):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = cfg
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg must convert to a dict, got {type(cfg_dict)}")

    trainer_name, trainer_params = validate_trainer_config(cfg_dict)
    trainer_cls = modules["get_trainer"](trainer_name)

    # Extract special parameters
    device_param = trainer_params.pop("device", device)
    final_device = device_param if device_param else device
    artifact_dumper = modules.get("artifact_dumper", None)

    trainer_name_lower = trainer_name.lower()

    # === Standard Trainer ===
    if trainer_name_lower in ["supervised", "autoencoder", "msn"]:
        trainer = setup_standard_trainer(
            trainer_cls,
            model,
            train_loader,
            val_loader,
            loss_fn,
            scheduler,
            final_device,
            artifact_dumper,
            trainer_params,
            cfg_dict,
        )

    # === GAN Trainer ===
    elif trainer_name_lower == "gan":
        trainer = setup_gan_trainer(
            trainer_cls,
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            scheduler,
            final_device,
            artifact_dumper,
            trainer_params,
            save_dir,
        )

    # === Fusion Trainer ===
    elif trainer_name_lower == "fusion":
        trainer = setup_fusion_trainer(
            trainer_cls,
            model,
            train_loader,
            val_loader,
            final_device,
            artifact_dumper,
            trainer_params,
            cfg_dict,
        )

    # === Fallback Trainer ===
    else:
        trainer = setup_fallback_trainer(
            trainer_cls,
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            scheduler,
            final_device,
            artifact_dumper,
            trainer_params,
        )

    return trainer
