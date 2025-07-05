"""
Trainer builder for Refrakt.

This module provides utilities to construct and initialize trainers from configuration dictionaries.
It supports standard, GAN, and fusion trainers, and ensures robust type checking and flexible parameter handling.

Typical usage involves passing a configuration (OmegaConf), model, dataloaders, loss, optimizer, scheduler, and other components to build a trainer for training or evaluation.
"""

from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from refrakt_core.registry.trainer_registry import register_trainer
from .utils.trainer_utils import (
    validate_trainer_config,
    setup_standard_trainer,
    setup_gan_trainer,
    setup_fusion_trainer,
    setup_fallback_trainer,
)

@register_trainer('ml')
class DummyMLTrainer:
    def __init__(self, *args, **kwargs):
        pass
    def train(self, *args, **kwargs):
        pass
    def evaluate(self, *args, **kwargs):
        pass


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

    This function supports standard, GAN, and fusion trainers, and ensures all parameters are type-checked and compatible with the training pipeline.
    It handles special cases for GAN and fusion trainers, including artifact dumping and fusion head construction.

    Args:
        cfg (OmegaConf): Configuration specifying the trainer type and parameters.
        model (Any): The model to be trained.
        train_loader (Any): DataLoader for training data.
        val_loader (Any): DataLoader for validation data.
        loss_fn (Any): Loss function or dictionary of loss functions.
        optimizer (Any): Optimizer or dictionary of optimizers.
        scheduler (Any): Learning rate scheduler or dictionary of schedulers.
        device (str): Device on which to run the trainer.
        modules (Dict[str, Any]): Registry of available trainer, artifact, and utility functions.
        save_dir (Optional[str]): Directory to save checkpoints and artifacts.

    Returns:
        Any: Instantiated trainer object, ready for training or evaluation.

    Raises:
        TypeError: If the configuration or its fields are not of the expected type.
        ValueError: If required trainer or fusion components are missing or not found in the registry.
    """
    print("Initializing trainer...")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
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
            trainer_cls, model, train_loader, val_loader, loss_fn, optimizer, scheduler,
            final_device, artifact_dumper, trainer_params, cfg_dict
        )

    # === GAN Trainer ===
    elif trainer_name_lower == "gan":
        trainer = setup_gan_trainer(
            trainer_cls, model, train_loader, val_loader, loss_fn, optimizer, scheduler,
            final_device, artifact_dumper, trainer_params, save_dir
        )

    # === Fusion Trainer ===
    elif trainer_name_lower == "fusion":
        trainer = setup_fusion_trainer(
            trainer_cls, model, train_loader, val_loader, final_device, artifact_dumper,
            trainer_params, cfg_dict
        )

    # === Fallback Trainer ===
    else:
        trainer = setup_fallback_trainer(
            trainer_cls, model, train_loader, val_loader, loss_fn, optimizer, scheduler,
            final_device, artifact_dumper, trainer_params
        )

    return trainer
