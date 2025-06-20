# trainer_builder.py

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
    """Initialize trainer based on configuration."""
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
    artifact_dumper = modules.get("artifact_dumper", None)

    trainer_name = cfg.trainer.name.lower()

    # === Standard Trainer ===
    if trainer_name in ["supervised", "autoencoder", "msn"]:
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
            artifact_dumper=artifact_dumper,
            **trainer_params,
        )

    # === GAN Trainer ===
    elif trainer_name == "gan":
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

    elif trainer_name == "fusion":
        from refrakt_core.integrations.sklearn.wrapper import SklearnWrapper

        fusion_cfg = cfg.model.get("fusion")
        if fusion_cfg is None:
            raise ValueError("[ERROR] 'model.fusion' block is required for FusionTrainer.")

        if fusion_cfg.type != "sklearn":
            raise ValueError(f"[ERROR] Unsupported fusion type: {fusion_cfg.type}")

        fusion_head = SklearnWrapper(
            fusion_cfg.model,
            **OmegaConf.to_container(fusion_cfg.get("params", {}), resolve=True),
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
