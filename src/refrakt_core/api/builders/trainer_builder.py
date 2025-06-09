"""# trainer_builder.py"""
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
    modules: Dict,
    save_dir: Optional[str],
) -> Any:
    """Initialize trainer based on configuration"""
    print("Initializing trainer...")
    trainer_cls = modules["get_trainer"](cfg.trainer.name)
    trainer_params = (
        OmegaConf.to_container(cfg.trainer.params, resolve=True)
        if cfg.trainer.params
        else {}
    )

    # Extract special parameters
    device_param = trainer_params.pop("device", device)
    final_device = device_param if device_param else device

    # Handle different trainer types
    if cfg.trainer.name != "gan":
        # For supervised trainers, pass optimizer class and arguments
        opt_map = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
            "rmsprop": torch.optim.RMSprop,
        }
        opt_cls = opt_map.get(cfg.optimizer.name.lower())
        optimizer_params = cfg.optimizer.params or {}

        trainer = trainer_cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer_cls=opt_cls,
            optimizer_args=optimizer_params,
            device=final_device,
            scheduler=scheduler,
            **trainer_params,
        )
    else:
        # For other trainers (GAN, etc.), pass optimizer instance or dict
        if "save_dir" in trainer_params:
            trainer_params.pop("save_dir")
        trainer = trainer_cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=final_device,
            scheduler=scheduler,
            save_dir=save_dir,
            **trainer_params,
        )

    return trainer
