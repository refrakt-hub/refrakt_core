"""
Trainer utilities for Refrakt.

This module provides utility functions for trainer initialization and configuration,
extracted from the main trainer_builder to reduce complexity and improve \
    maintainability.

The module handles:
- Trainer configuration validation and parsing
- Standard trainer setup (supervised, autoencoder, msn)
- GAN trainer setup with generator/discriminator components
- Fusion trainer setup for ensemble models
- Fallback trainer setup for error handling
- Optimizer mapping and configuration

These utilities support various training architectures including standard supervised
training, GAN training with separate generator/discriminator optimizers, and fusion
training for ensemble models with sklearn components.

Typical usage involves passing configuration dictionaries and component objects
to build and configure trainers for different training scenarios.
"""

import typing
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf


def validate_trainer_config(cfg_dict: Any) -> tuple[str, Dict[str, Any]]:
    """
    Validate and extract trainer configuration parameters.

    This function parses the trainer configuration dictionary to extract the \
        trainer name and parameters. It performs type validation to ensure the \
        configuration structure is correct before trainer instantiation.

    Args:
        cfg_dict: Configuration dictionary containing trainer specifications.
                  Expected to have a 'trainer' key with 'name' and 'params' fields.

    Returns:
        Tuple containing:
        - trainer_name (str): The name of the trainer to instantiate
        - trainer_params (Dict[str, Any]): Trainer-specific parameters

    Raises:
        TypeError: If the configuration structure is invalid or required fields
                  are missing or of incorrect type.
    """
    trainer_cfg = cfg_dict.get("trainer")
    if not isinstance(trainer_cfg, dict):
        raise TypeError(f"cfg.trainer must be a dict, got {type(trainer_cfg)}")

    trainer_name = trainer_cfg.get("name")
    if not isinstance(trainer_name, str):
        raise TypeError(f"trainer name must be a str, got {type(trainer_name)}")

    trainer_params = trainer_cfg.get("params", {}) or {}
    if not isinstance(trainer_params, dict):
        raise TypeError(f"trainer_params must be a dict, got {type(trainer_params)}")

    return trainer_name, trainer_params


def setup_standard_trainer(
    trainer_cls: Any,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    loss_fn: Any,
    scheduler: Any,
    device: str,
    artifact_dumper: Optional[Any],
    trainer_params: Dict[str, Any],
    cfg_dict: Any,
) -> Any:
    """
    Setup a standard trainer for supervised, autoencoder, or MSN training.

    This function configures a standard trainer with all necessary components
    including model, data loaders, loss function, optimizer, and scheduler.
    It handles optimizer class mapping and parameter validation.

    Args:
        trainer_cls: Trainer class to instantiate
        model: Model object to train
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function for training
        scheduler: Learning rate scheduler
        device: Target device string (e.g., "cuda", "cpu")
        artifact_dumper: Optional artifact dumper for saving outputs
        trainer_params: Trainer-specific parameters dictionary
        cfg_dict: Configuration dictionary containing optimizer settings

    Returns:
        Configured trainer instance ready for training

    Raises:
        TypeError: If optimizer configuration is invalid or missing required fields
    """
    opt_map = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }

    optimizer_cfg = cfg_dict.get("optimizer")
    if not isinstance(optimizer_cfg, dict):
        raise TypeError(f"cfg.optimizer must be a dict, got {type(optimizer_cfg)}")

    opt_name = optimizer_cfg.get("name")
    if not isinstance(opt_name, str):
        raise TypeError(f"optimizer name must be a str, got {type(opt_name)}")

    opt_cls = opt_map.get(opt_name.lower())
    optimizer_params = optimizer_cfg.get("params", {}) or {}
    if not isinstance(optimizer_params, dict):
        raise TypeError(
            f"optimizer_params must be a dict, got {type(optimizer_params)}"
        )

    return trainer_cls(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer_cls=opt_cls,
        optimizer_args=optimizer_params,
        device=device,
        scheduler=scheduler,
        artifact_dumper=artifact_dumper,
        **trainer_params,
    )


def setup_gan_trainer(
    trainer_cls: Any,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    loss_fn: Any,
    optimizer: Any,
    scheduler: Any,
    device: str,
    artifact_dumper: Optional[Any],
    trainer_params: Dict[str, Any],
    save_dir: Optional[str],
) -> Any:
    """
    Setup a GAN trainer for generator/discriminator training.

    This function configures a GAN trainer with separate loss functions and
    optimizers for generator and discriminator components. It handles the
    special requirements of GAN training architectures.

    Args:
        trainer_cls: GAN trainer class to instantiate
        model: Model object containing generator and discriminator
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Dictionary of loss functions for generator and discriminator
        optimizer: Dictionary of optimizers for generator and discriminator
        scheduler: Learning rate scheduler
        device: Target device string (e.g., "cuda", "cpu")
        artifact_dumper: Optional artifact dumper for saving outputs
        trainer_params: Trainer-specific parameters dictionary
        save_dir: Directory to save model checkpoints

    Returns:
        Configured GAN trainer instance ready for training

    Note:
        The loss_fn and optimizer parameters are expected to be dictionaries
        with 'generator' and 'discriminator' keys for GAN architectures.
    """
    if "save_dir" in trainer_params:
        trainer_params.pop("save_dir")

    return trainer_cls(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,  # Dict of loss wrappers
        optimizer_cls=optimizer,  # Dict of optimizers
        device=device,
        scheduler=scheduler,
        artifact_dumper=artifact_dumper,
        save_dir=save_dir,
        **trainer_params,
    )


def setup_fusion_trainer(
    trainer_cls: Any,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    device: str,
    artifact_dumper: Optional[Any],
    trainer_params: Dict[str, Any],
    cfg_dict: Any,
) -> Any:
    """
    Setup a fusion trainer for ensemble model training.

    This function configures a fusion trainer for models that combine multiple
    components or modalities. It validates fusion configuration and ensures
    proper setup for ensemble training scenarios.

    Args:
        trainer_cls: Fusion trainer class to instantiate
        model: Model object containing fusion components
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Target device string (e.g., "cuda", "cpu")
        artifact_dumper: Optional artifact dumper for saving outputs
        trainer_params: Trainer-specific parameters dictionary
        cfg_dict: Configuration dictionary containing fusion settings

    Returns:
        Configured fusion trainer instance ready for training

    Raises:
        TypeError: If model configuration is invalid
        ValueError: If fusion configuration is missing or invalid
    """
    from refrakt_core.integrations.cpu.wrapper import SklearnWrapper

    model_cfg = cfg_dict.get("model")
    if not isinstance(model_cfg, dict):
        raise TypeError(f"cfg.model must be a dict, got {type(model_cfg)}")

    fusion_cfg = model_cfg.get("fusion")
    if fusion_cfg is None or not isinstance(fusion_cfg, dict):
        raise ValueError("[ERROR] 'model.fusion' block is required for FusionTrainer.")

    fusion_type = fusion_cfg.get("type")
    if fusion_type != "sklearn":
        raise ValueError(f"[ERROR] Unsupported fusion type: {fusion_type}")

    fusion_params = OmegaConf.to_container(fusion_cfg.get("params", {}), resolve=True)
    if not isinstance(fusion_params, dict) or not all(
        isinstance(k, str) for k in fusion_params.keys()
    ):
        keys = list(fusion_params.keys()) if isinstance(fusion_params, dict) else "N/A"
        keys_str = f"{keys}"
        msg = (
            "fusion_params must be a dict with str keys, got "
            f"{type(fusion_params)} "
            "and keys "
            f"{keys_str}"
        )
        raise TypeError(msg)
    fusion_params = typing.cast(Dict[str, Any], fusion_params)

    model_name = fusion_cfg.get("model")
    if not isinstance(model_name, str):
        raise TypeError(f"fusion_cfg['model'] must be a str, got {type(model_name)}")

    fusion_head = SklearnWrapper(model_name, **fusion_params)

    return trainer_cls(
        model=model,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        artifact_dumper=artifact_dumper,
        **trainer_params,
    )


def setup_fallback_trainer(
    trainer_cls: Any,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    loss_fn: Any,
    optimizer: Any,
    scheduler: Any,
    device: str,
    artifact_dumper: Optional[Any],
    trainer_params: Dict[str, Any],
) -> Any:
    """
    Setup a fallback trainer for unknown trainer types.

    Args:
        trainer_cls: Trainer class
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        artifact_dumper: Artifact dumper
        trainer_params: Trainer parameters

    Returns:
        Configured trainer instance
    """
    return trainer_cls(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        artifact_dumper=artifact_dumper,
        **trainer_params,
    )
