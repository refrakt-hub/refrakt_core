import gc
import os
import sys
import traceback
from typing import Optional

import torch
from omegaconf import OmegaConf

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.builders.loss_builder import build_loss
from refrakt_core.logging import get_global_logger
from refrakt_core.schema.artifact import ArtifactDumper

import refrakt_core.models
import refrakt_core.losses
import refrakt_core.trainer
import refrakt_core.datasets
import refrakt_core.transforms
import refrakt_core.wrappers

gc.collect()
torch.cuda.empty_cache()

def train(
    config_path: str,
    model_path: Optional[str] = None,
    logger: Optional[RefraktLogger] = None,
):
    from refrakt_core.registry.loss_registry import get_loss
    from refrakt_core.registry.model_registry import get_model
    from refrakt_core.registry.trainer_registry import get_trainer

    try:
        cfg = OmegaConf.load(config_path)

        # === Logger Setup ===
        if logger is None:
            log_types = cfg.logger.get("log_type", [])
            valid_log_types = {"wandb", "tensorboard"}
            unknown = set(log_types) - valid_log_types
            if unknown:
                raise ValueError(f"❌ Unsupported log_type(s): {unknown}. Supported: {valid_log_types}")

            logger = RefraktLogger(
                model_name=cfg.model.name,
                log_dir=cfg.logger.get("log_dir", "./logs"),
                log_types=log_types,
                console=cfg.logger.get("console", True),
                debug=cfg.logger.get("debug", False),
            )

        logger.log_config(OmegaConf.to_container(cfg, resolve=True))

        # === Device Setup ===
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # === Dataset ===
        logger.info("Building datasets...")
        train_dataset = build_dataset(cfg.dataset)
        val_cfg = OmegaConf.merge(cfg.dataset, OmegaConf.create({"params": {"train": False}}))
        val_dataset = build_dataset(val_cfg)

        logger.info("Building data loaders...")
        train_loader = build_dataloader(train_dataset, cfg.dataloader)
        val_loader = build_dataloader(val_dataset, cfg.dataloader)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # === Model ===
        logger.info("Building model...")
        model = build_model(cfg, modules={"get_model": get_model}, device=device)

        # === Loss ===
        logger.info("Building loss function...")
        loss_fn = build_loss(cfg, modules={"get_loss": get_loss}, device=device)

        # === Optimizer ===
        logger.info("Building optimizer...")
        opt_map = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
            "rmsprop": torch.optim.RMSprop,
        }

        if cfg.optimizer.get("generator") or cfg.optimizer.get("discriminator"):
            optimizer = {}
            for comp_name in ["generator", "discriminator"]:
                comp_cfg = cfg.optimizer.get(comp_name)
                if comp_cfg:
                    opt_name = comp_cfg["name"]
                    opt_cls = opt_map.get(opt_name.lower())
                    if not opt_cls:
                        raise ValueError(f"Unsupported optimizer for {comp_name}: {opt_name}")
                    opt_params = comp_cfg.get("params", {})
                    params = getattr(model, comp_name).parameters()
                    optimizer[comp_name] = opt_cls(params, **opt_params)
                    logger.info(f"Optimizer ({comp_name}): {opt_name} with params: {opt_params}")
        elif cfg.optimizer.get("components"):
            optimizer = {}
            for comp_name, comp_cfg in cfg.optimizer.components.items():
                opt_name = comp_cfg["name"]
                opt_cls = opt_map.get(opt_name.lower())
                if not opt_cls:
                    raise ValueError(f"Unsupported optimizer for {comp_name}: {opt_name}")
                opt_params = comp_cfg.get("params", {})
                params = getattr(model, comp_name).parameters()
                optimizer[comp_name] = opt_cls(params, **opt_params)
                logger.info(f"Optimizer ({comp_name}): {opt_name} with params: {opt_params}")
        else:
            opt_cls = opt_map.get(cfg.optimizer.name.lower())
            if not opt_cls:
                raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")
            optimizer_params = cfg.optimizer.params or {}
            optimizer = opt_cls(model.parameters(), **optimizer_params)
            logger.info(f"Optimizer: {cfg.optimizer.name} with params: {optimizer_params}")

        # === Scheduler ===
        scheduler = None
        if cfg.scheduler and cfg.scheduler.name:
            logger.info("Building scheduler...")
            sched_map = {
                "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
                "steplr": torch.optim.lr_scheduler.StepLR,
                "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
                "exponential": torch.optim.lr_scheduler.ExponentialLR,
            }
            scheduler_cls = sched_map.get(cfg.scheduler.name.lower())
            if not scheduler_cls:
                raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")
            scheduler_params = cfg.scheduler.params or {}
            scheduler = scheduler_cls(optimizer, **scheduler_params)
            logger.info(f"Scheduler: {cfg.scheduler.name} with params: {scheduler_params}")

        # === Trainer ===
        logger.info("Initializing trainer...")
        trainer_cls = get_trainer(cfg.trainer.name)

        trainer_params = OmegaConf.to_container(cfg.trainer.params, resolve=True) if cfg.trainer.params else {}
        if cfg.model.name == "autoencoder":
            variant = cfg.model.params.get("type", "simple")
            trainer_params["model_variant"] = variant

        num_epochs = trainer_params.pop("num_epochs", 1)
        device_param = trainer_params.pop("device", device)
        final_device = device_param or device
        trainer_params["logger"] = logger

        # === ArtifactDumper (logs every 1 batches) ===
        artifact_dumper = ArtifactDumper(enabled=True, model_name=model, base_path="./artifacts/train")
        artifact_dumper.log_every = 1 # Optional future config
        trainer_params["artifact_dumper"] = artifact_dumper

        if cfg.trainer.name != "gan":
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
            if not hasattr(model, "generator"):
                raise ValueError("GAN trainer requires model to have a 'generator'")
            trainer = trainer_cls(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=final_device,
                scheduler=scheduler,
                **trainer_params,
            )

        # === Train ===
        logger.info(f"\nStarting training for {num_epochs} epochs...")
        trainer.train(num_epochs=num_epochs)

        logger.info("Saving model now...")
        trainer.save(path=model_path)

        # Save config alongside model
        config_save_path = os.path.join(trainer.save_dir or "./checkpoints", f"{trainer.model.__class__.__name__}.yaml")
        OmegaConf.save(cfg, config_save_path)
        logger.info(f"Saved config to {config_save_path}")

        logger.info("\n✅ Training completed successfully!")

    except Exception as e:
        logger = logger or get_global_logger()
        logger.error(f"\n❌ Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
