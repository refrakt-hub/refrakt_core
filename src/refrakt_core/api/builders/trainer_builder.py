"""
Trainer builder for Refrakt.

This module provides utilities to construct and initialize trainers from configuration dictionaries.
It supports standard, GAN, and fusion trainers, and ensures robust type checking and flexible parameter handling.

Typical usage involves passing a configuration (OmegaConf), model, dataloaders, loss, optimizer, scheduler, and other components to build a trainer for training or evaluation.
"""

import typing
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf


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
    trainer_cfg = cfg_dict.get("trainer")
    if not isinstance(trainer_cfg, dict):
        raise TypeError(f"cfg.trainer must be a dict, got {type(trainer_cfg)}")
    trainer_name = trainer_cfg.get("name")
    if not isinstance(trainer_name, str):
        raise TypeError(f"trainer name must be a str, got {type(trainer_name)}")
    trainer_cls = modules["get_trainer"](trainer_name)
    trainer_params = trainer_cfg.get("params", {}) or {}
    if not isinstance(trainer_params, dict):
        raise TypeError(f"trainer_params must be a dict, got {type(trainer_params)}")

    # Extract special parameters
    device_param = trainer_params.pop("device", device)
    final_device = device_param if device_param else device
    artifact_dumper = modules.get("artifact_dumper", None)

    trainer_name_lower = trainer_name.lower()

    # === Standard Trainer ===
    if trainer_name_lower in ["supervised", "autoencoder", "msn"]:
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

        trainer = trainer_cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer_cls=opt_cls,
            optimizer_args=optimizer_params,
            device=final_device,
            scheduler=scheduler,
            artifact_dumper=artifact_dumper,
            **trainer_params,
        )

    # === GAN Trainer ===
    elif trainer_name_lower == "gan":
        if "save_dir" in trainer_params:
            trainer_params.pop("save_dir")

        trainer = trainer_cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,  # Dict of loss wrappers
            optimizer_cls=optimizer,  # Dict of optimizers
            device=final_device,
            scheduler=scheduler,
            artifact_dumper=artifact_dumper,
            save_dir=save_dir,
            **trainer_params,
        )

    elif trainer_name_lower == "fusion":
        from refrakt_core.integrations.sklearn.wrapper import SklearnWrapper

        model_cfg = cfg_dict.get("model")
        if not isinstance(model_cfg, dict):
            raise TypeError(f"cfg.model must be a dict, got {type(model_cfg)}")
        fusion_cfg = model_cfg.get("fusion")
        if fusion_cfg is None or not isinstance(fusion_cfg, dict):
            raise ValueError(
                "[ERROR] 'model.fusion' block is required for FusionTrainer."
            )

        fusion_type = fusion_cfg.get("type")
        if fusion_type != "sklearn":
            raise ValueError(f"[ERROR] Unsupported fusion type: {fusion_type}")

        fusion_params = OmegaConf.to_container(
            fusion_cfg.get("params", {}), resolve=True
        )
        if not isinstance(fusion_params, dict) or not all(
            isinstance(k, str) for k in fusion_params.keys()
        ):
            raise TypeError(
                f"fusion_params must be a dict with str keys, got {type(fusion_params)} and keys {list(fusion_params.keys()) if isinstance(fusion_params, dict) else 'N/A'}"
            )
        fusion_params = typing.cast(Dict[str, Any], fusion_params)
        model_name = fusion_cfg.get("model")
        if not isinstance(model_name, str):
            raise TypeError(
                f"fusion_cfg['model'] must be a str, got {type(model_name)}"
            )
        fusion_head = SklearnWrapper(
            model_name,
            **fusion_params,
        )

        trainer = trainer_cls(
            model=model,
            fusion_head=fusion_head,
            train_loader=train_loader,
            val_loader=val_loader,
            device=final_device,
            artifact_dumper=artifact_dumper,
            **trainer_params,
        )

    # === Fallback Trainer ===
    else:
        trainer = trainer_cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=final_device,
            scheduler=scheduler,
            artifact_dumper=artifact_dumper,
            **trainer_params,
        )

    return trainer
