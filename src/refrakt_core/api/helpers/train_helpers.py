"""
Helper functions for the train API.

This module contains internal helper functions used by the main train function.
"""

import torch
from typing import Any, Dict, Optional, cast, Union
from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.train_utils import (
    load_config,
    _resolve_model_name_train,
    _handle_pure_ml_training,
    _setup_optimizer_config,
    _setup_trainer_params,
    _handle_fusion_training,
    _save_config_and_log_metrics
)


def _load_and_validate_config(cfg: Union[str, DictConfig]) -> DictConfig:
    """Load and validate configuration."""
    if isinstance(cfg, str):
        config = load_config(cfg)
    else:
        config = cfg
    return config


def _setup_logging(config: DictConfig, resolved_model_name: str, logger: Optional[RefraktLogger]) -> RefraktLogger:
    """Setup logging configuration."""
    from refrakt_core.api.utils.train_utils import setup_logger
    
    if logger is None:
        logger = setup_logger(config, resolved_model_name)
    
    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a dict after OmegaConf.to_container.")
    logger.log_config(cast(Dict[str, Any], config_dict))
    return logger


def _check_pure_ml_training(config: DictConfig) -> bool:
    """Check if this is a pure ML training session."""
    return (getattr(config.model, 'type', None) == 'ml' or 
            getattr(config.dataset, 'name', None) == 'tabular_ml')


def _get_modules_and_device() -> tuple[Dict[str, Any], torch.device]:
    """Get registry modules and device."""
    from refrakt_core.registry.loss_registry import get_loss
    from refrakt_core.registry.model_registry import get_model
    from refrakt_core.registry.trainer_registry import get_trainer
    from refrakt_core.registry.wrapper_registry import get_wrapper

    modules = {
        "get_model": get_model,
        "get_loss": get_loss,
        "get_trainer": get_trainer,
        "get_wrapper": get_wrapper,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return modules, device


def _build_datasets_and_model(config: DictConfig, modules: Dict[str, Any], device: torch.device, logger: RefraktLogger):
    """Build datasets, loaders, and model."""
    from refrakt_core.api.utils.train_utils import build_datasets_and_loaders_with_resize
    from refrakt_core.api.builders.model_builder import build_model
    from refrakt_core.api.builders.loss_builder import build_loss

    train_dataset, val_dataset, train_loader, val_loader = build_datasets_and_loaders_with_resize(config, logger)
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

    loss_fn = build_loss(
        cast(OmegaConf, config), modules=modules, device=str(device)
    )

    return train_loader, val_loader, model, loss_fn


def _setup_optimizer_and_scheduler(config: DictConfig, model: torch.nn.Module):
    """Setup optimizer and scheduler."""
    opt_cls, optimizer_args = _setup_optimizer_config(config)
    optimizer = opt_cls(model.parameters(), **optimizer_args)

    scheduler = None
    if hasattr(config, "scheduler") and config.scheduler:
        from refrakt_core.api.builders.scheduler_builder import build_scheduler
        scheduler = build_scheduler(cast(OmegaConf, config), optimizer)

    return optimizer, scheduler


def _setup_trainer(config: DictConfig, model: torch.nn.Module, train_loader: Any, val_loader: Any, 
                   loss_fn: Any, optimizer: Any, scheduler: Any, device: str, modules: Dict[str, Any], 
                   artifact_dumper: Any, resolved_model_name: str, logger: RefraktLogger):
    """Setup and initialize trainer."""
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


def _execute_training(trainer: Any, num_epochs: int, config: DictConfig, model: Any, 
                     train_loader: Any, val_loader: Any, final_device: str, 
                     artifact_dumper: Any, resolved_model_name: str, logger: RefraktLogger):
    """Execute the training process."""
    logger.info(f"\n🚀 Starting training for {num_epochs} epochs...")
    final_metrics = trainer.train(num_epochs)

    # Fusion Head Training
    _handle_fusion_training(config, model, train_loader, val_loader, final_device, artifact_dumper, trainer, logger)

    # Save config and log final metrics
    _save_config_and_log_metrics(config, trainer, resolved_model_name, final_metrics, logger)

    return final_metrics 