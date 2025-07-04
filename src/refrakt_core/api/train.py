"""
Training entry point for Refrakt.

This module orchestrates the training pipeline using utility functions for config, logger, model, data, and artifact handling.
"""

import gc
import os
import sys
import traceback
from typing import Any, Dict, Optional, cast

import torch
from omegaconf import OmegaConf
import refrakt_core.models
import refrakt_core.wrappers
import refrakt_core.losses
import refrakt_core.trainer
import refrakt_core.datasets
import refrakt_core.transforms
import refrakt_core.registry
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.train_utils import (build_datasets_and_loaders,
                                                build_model_and_log_graph,
                                                build_optimizer_and_scheduler,
                                                load_config,
                                                setup_artifact_dumper,
                                                setup_logger)
from refrakt_core.global_logging import get_global_logger
from refrakt_core.integrations.fusion.builder import build_fusion_head
from refrakt_core.integrations.fusion.trainer import FusionTrainer

gc.collect()
torch.cuda.empty_cache()

import warnings

warnings.filterwarnings("ignore")


def train(
    config_path: str,
    model_path: Optional[str] = None,
    logger: Optional[RefraktLogger] = None,
) -> None:
    """
    Orchestrate the training pipeline for Refrakt.

    Args:
        config_path (str): Path to the configuration file.
        model_path (Optional[str]): Path to save the trained model.
        logger (Optional[RefraktLogger]): Logger instance (optional).
    """
    try:
        cfg = load_config(config_path)
        # Resolve model name
        if cfg.model.name == "autoencoder":
            variant = cfg.model.params.get("variant", "simple")
            resolved_model_name = f"autoencoder_{variant}"
        else:
            resolved_model_name = cfg.model.name

        # Logger
        if logger is None:
            logger = setup_logger(cfg, resolved_model_name)
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(config_dict, dict):
            raise TypeError("Config must be a dict after OmegaConf.to_container.")
        logger.log_config(cast(Dict[str, Any], config_dict))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Data
        train_dataset, val_dataset, train_loader, val_loader = (
            build_datasets_and_loaders(cfg)
        )
        logger.info(
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )

        # Model
        from refrakt_core.registry.model_registry import get_model
        from refrakt_core.registry.wrapper_registry import get_wrapper

        modules = {"get_model": get_model, "get_wrapper": get_wrapper}
        model = build_model_and_log_graph(cfg, modules, device, train_loader, logger)
        logger.info(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        # Loss, Optimizer, Scheduler
        from refrakt_core.api.builders.loss_builder import build_loss
        from refrakt_core.registry.loss_registry import get_loss

        loss_fn = build_loss(
            cast(OmegaConf, cfg), modules={"get_loss": get_loss}, device=device
        )
        optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)

        # --- Extract optimizer class and args ---
        from torch import optim
        opt_map = {
            "adam": optim.Adam,
            "sgd": optim.SGD,
            "adamw": optim.AdamW,
            "rmsprop": optim.RMSprop,
        }
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise TypeError(f"Config must be a dict after OmegaConf.to_container, got {type(cfg_dict)}")
        optimizer_cfg = cfg_dict.get("optimizer", {})
        opt_name = optimizer_cfg.get("name", "adamw")
        opt_cls = opt_map[opt_name.lower()]
        optimizer_args = optimizer_cfg.get("params", {}) or {}

        # Artifact Dumper
        artifact_dumper = setup_artifact_dumper(cfg, resolved_model_name, logger)

        # Trainer
        from refrakt_core.registry.trainer_registry import get_trainer

        trainer_cls = get_trainer(cfg.trainer.name)
        trainer_params = (
            OmegaConf.to_container(cfg.trainer.params, resolve=True)
            if cfg.trainer.params
            else {}
        )
        if not isinstance(trainer_params, dict):
            trainer_params = {}
        trainer_params = cast(Dict[str, Any], trainer_params)
        num_epochs = trainer_params.pop("num_epochs", 1)
        device_param = trainer_params.pop("device", device)
        final_device = device_param or device
        trainer_params["logger"] = logger
        trainer_params["artifact_dumper"] = artifact_dumper
        trainer = trainer_cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer_cls=opt_cls,
            optimizer_args=optimizer_args,
            device=final_device,
            scheduler=scheduler,
            **trainer_params,
        )
        trainer.model_name = resolved_model_name

        # Train
        logger.info(f"\nStarting training for {num_epochs} epochs...")
        final_metrics = trainer.train(num_epochs=num_epochs)
        logger.info("Saving model now...")
        trainer.save(path=model_path)

        # Fusion Support
        if hasattr(cfg.model, "fusion"):
            logger.info(
                "\n[FUSION] Fusion head config detected. Starting fusion head training..."
            )
            fusion_cfg = cfg.model.fusion
            fusion_head = build_fusion_head(fusion_cfg)
            fusion_trainer = FusionTrainer(
                model=model,
                fusion_head=fusion_head,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                artifact_dumper=artifact_dumper,
                model_name=cfg.trainer.params.model_name,
            )
            fusion_metrics = fusion_trainer.train()
            fusion_save_path = os.path.join(
                cfg.trainer.params.save_dir,
                f"{cfg.trainer.params.model_name}_fusion.joblib",
            )
            save_method = getattr(fusion_head, "save", None)
            if callable(save_method):
                save_method(fusion_save_path)
                logger.info(f"[FUSION] Fusion head saved to {fusion_save_path}")
            if logger:
                logger.log_metrics(
                    fusion_metrics, step=trainer.global_step, prefix="fusion"
                )

        # Save Config
        config_save_path = os.path.join(
            getattr(trainer, "save_dir", os.path.join("./artifacts", "yaml")),
            f"{resolved_model_name}.yaml",
        )
        OmegaConf.save(cfg, config_save_path)
        logger.info(f"Saved config to {config_save_path}")

        # Log Final Metrics
        print("\nFinal Metrics:", final_metrics)
        if logger:
            logger.log_metrics(final_metrics, step=trainer.global_step, prefix="final")
        logger.info("\n✅ Training completed successfully!")

    except Exception as e:
        # Ensure logger is a RefraktLogger or None
        if not (logger and hasattr(logger, "error")):
            logger = None
        if logger:
            logger.error(f"\n❌ Training failed: {str(e)}")
            logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        gc.collect()
        torch.cuda.empty_cache()
