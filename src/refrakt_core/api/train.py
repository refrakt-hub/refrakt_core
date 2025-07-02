import gc
import os
import sys
import traceback
from typing import Optional

import torch
from omegaconf import OmegaConf

import refrakt_core.datasets
import refrakt_core.losses
import refrakt_core.models
import refrakt_core.trainer
import refrakt_core.transforms
import refrakt_core.wrappers
from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.loss_builder import build_loss
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.global_logging import get_global_logger
from refrakt_core.integrations.fusion.builder import build_fusion_head
from refrakt_core.integrations.sklearn.trainer import FusionTrainer
from refrakt_core.schema.artifact import ArtifactDumper

gc.collect()
torch.cuda.empty_cache()

import warnings

warnings.filterwarnings("ignore")


def train(
    config_path: str,
    model_path: Optional[str] = None,
    logger: Optional[RefraktLogger] = None,
):
    from refrakt_core.registry.loss_registry import get_loss
    from refrakt_core.registry.model_registry import get_model
    from refrakt_core.registry.trainer_registry import get_trainer
    from refrakt_core.registry.wrapper_registry import get_wrapper

    try:
        cfg = OmegaConf.load(config_path)

        if cfg.model.name == "autoencoder":
            variant = cfg.model.params.get("variant", "simple")
            resolved_model_name = f"autoencoder_{variant}"
        else:
            resolved_model_name = cfg.model.name

        # === Logger Setup ===
        if logger is None:
            runtime_cfg = cfg.get("runtime", {})
            log_types = runtime_cfg.get("log_type", [])
            valid_log_types = {"wandb", "tensorboard"}
            unknown = set(log_types) - valid_log_types
            if unknown:
                raise ValueError(
                    f"❌ Unsupported log_type(s): {unknown}. Supported: {valid_log_types}"
                )

            logger = RefraktLogger(
                model_name=resolved_model_name,
                log_dir=runtime_cfg.get("log_dir", "./logs"),
                log_types=log_types,
                console=runtime_cfg.get("console", True),
                debug=runtime_cfg.get("debug", False),
            )

        # Log config early for WandB
        logger.log_config(OmegaConf.to_container(cfg, resolve=True))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # === Dataset + Dataloader ===
        logger.info("Building datasets...")
        train_dataset = build_dataset(cfg.dataset)
        val_cfg = OmegaConf.merge(
            cfg.dataset, OmegaConf.create({"params": {"train": False}})
        )
        val_dataset = build_dataset(val_cfg)

        logger.info("Building data loaders...")
        train_loader = build_dataloader(train_dataset, cfg.dataloader)
        val_loader = build_dataloader(val_dataset, cfg.dataloader)
        logger.info(
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )

        # === Model ===
        logger.info("Building model...")
        model_cls = get_model(cfg.model.name)
        model = build_model(
            cfg,
            modules={
                "get_model": get_model,
                "get_wrapper": get_wrapper,
                "model": model_cls,
            },
            device=device,
        )

        # === Log Model Graph ===
        import torch.nn.functional as F

        logger.info("Logging model graph...")
        try:
            if cfg.model.name == "dino":
                raise NotImplementedError("Graph logging for DINO is not yet supported")

            sample_batch = next(iter(train_loader))
            if isinstance(sample_batch, (tuple, list)):
                sample_input = sample_batch[0]
            else:
                sample_input = sample_batch

            # ✅ Safely move to device
            if isinstance(sample_input, dict):
                sample_input = {k: v.to(device) for k, v in sample_input.items()}
            else:
                sample_input = sample_input.to(device)
            logger.log_model_graph(model, sample_input)
        except Exception as e:
            logger.error(f"Model graph logging failed: {str(e)}")

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Trainable parameters: {count_parameters(model):,}")

        # === Loss Function ===
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

        optimizer = None
        opt_cls_used = None
        opt_params = None

        if cfg.optimizer.get("generator") or cfg.optimizer.get("discriminator"):
            optimizer = {}
            for comp_name in ["generator", "discriminator"]:
                comp_cfg = cfg.optimizer.get(comp_name)
                if comp_cfg:
                    opt_name = comp_cfg["name"]
                    opt_cls = opt_map.get(opt_name.lower())
                    if not opt_cls:
                        raise ValueError(
                            f"Unsupported optimizer for {comp_name}: {opt_name}"
                        )
                    opt_params = comp_cfg.get("params", {})
                    params = getattr(model, comp_name).parameters()
                    optimizer[comp_name] = opt_cls(params, **opt_params)
                    logger.info(
                        f"Optimizer ({comp_name}): {opt_name} with params: {opt_params}"
                    )
        elif cfg.optimizer.get("components"):
            optimizer = {}
            for comp_name, comp_cfg in cfg.optimizer.components.items():
                opt_name = comp_cfg["name"]
                opt_cls = opt_map.get(opt_name.lower())
                if not opt_cls:
                    raise ValueError(
                        f"Unsupported optimizer for {comp_name}: {opt_name}"
                    )
                opt_params = comp_cfg.get("params", {})
                params = getattr(model, comp_name).parameters()
                optimizer[comp_name] = opt_cls(params, **opt_params)
                logger.info(
                    f"Optimizer ({comp_name}): {opt_name} with params: {opt_params}"
                )
        else:
            opt_name = cfg.optimizer.name
            opt_cls = opt_map.get(opt_name.lower())
            if not opt_cls:
                raise ValueError(f"Unsupported optimizer: {opt_name}")
            opt_params = cfg.optimizer.params or {}
            optimizer = opt_cls(model.parameters(), **opt_params)
            opt_cls_used = opt_cls
            logger.info(f"Optimizer: {opt_name} with params: {opt_params}")

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
            logger.info(
                f"Scheduler: {cfg.scheduler.name} with params: {scheduler_params}"
            )

        # === Trainer Setup ===
        logger.info("Initializing trainer...")
        trainer_cls = get_trainer(cfg.trainer.name)
        trainer_params = (
            OmegaConf.to_container(cfg.trainer.params, resolve=True)
            if cfg.trainer.params
            else {}
        )

        if cfg.model.name == "autoencoder":
            variant = cfg.model.params.get("type", "simple")
            trainer_params["model_variant"] = variant

        num_epochs = trainer_params.pop("num_epochs", 1)
        device_param = trainer_params.pop("device", device)
        final_device = device_param or device
        trainer_params["logger"] = logger

        # === Artifact Dumper ===
        # artifact_log_every = cfg.get("artifacts", {}).get("log_every", 1)
        # runtime_mode = cfg.get("runtime", {}).get("mode", "train")
        artifact_dumper = ArtifactDumper(
            enabled=True,
            model_name=resolved_model_name,
            base_path="./artifacts",
            log_every=1,
            logger=logger,
        )
        # artifact_dumper.log_every = artifact_log_every
        trainer_params["artifact_dumper"] = artifact_dumper

        # === Trainer Init ===
        if cfg.trainer.name != "gan":
            trainer = trainer_cls(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer_cls=opt_cls_used,
                optimizer_args=opt_params,
                device=final_device,
                scheduler=scheduler,
                **trainer_params,
            )
        else:
            # Build optimizer_cls dictionary for GAN
            optimizer_cls = {}
            optimizer_args = {}

            for comp_name in ["generator", "discriminator"]:
                comp_cfg = cfg.optimizer.get(comp_name)
                if not comp_cfg:
                    raise ValueError(f"Missing optimizer config for {comp_name}")

                opt_name = comp_cfg["name"]
                opt_cls = opt_map.get(opt_name.lower())
                if not opt_cls:
                    raise ValueError(
                        f"Unsupported optimizer for {comp_name}: {opt_name}"
                    )

                optimizer_cls[comp_name] = opt_cls
                optimizer_args = comp_cfg["params"]

            trainer = trainer_cls(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer_cls=optimizer_cls,
                optimizer_args=optimizer_args,
                device=final_device,
                scheduler=scheduler,
                **trainer_params,
            )

        trainer.model_name = resolved_model_name

        # === Train ===
        logger.info(f"\nStarting training for {num_epochs} epochs...")
        final_metrics = trainer.train(num_epochs=num_epochs)

        logger.info("Saving model now...")
        trainer.save(path=model_path)

        # === Fusion Support ===
        if "fusion" in cfg.model:
            logger.info("\n[FUSION] Welcome to True Fusion v1.")
            logger.info(
                "\n[FUSION] Fusion head config detected. Starting fusion head training..."
            )

            # Load fusion config
            fusion_cfg = cfg.model.fusion
            fusion_head = build_fusion_head(fusion_cfg)

            # Train using frozen backbone
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

            # Save the fusion model
            fusion_save_path = os.path.join(
                cfg.trainer.params.save_dir,
                f"{cfg.trainer.params.model_name}_fusion.joblib",
            )
            if hasattr(fusion_head, "save"):
                fusion_head.save(fusion_save_path)
                logger.info(f"[FUSION] Fusion head saved to {fusion_save_path}")

            # Log fusion metrics
            if logger:
                logger.log_metrics(
                    fusion_metrics, step=trainer.global_step, prefix="fusion"
                )

        # === Save Config ===
        config_save_path = os.path.join(
            trainer.save_dir or os.path.join("./artifacts", "yaml"),
            f"{resolved_model_name}.yaml",
        )
        OmegaConf.save(cfg, config_save_path)
        logger.info(f"Saved config to {config_save_path}")

        # === Log Final Metrics ===
        print("\nFinal Metrics:", final_metrics)
        if logger:
            logger.log_metrics(final_metrics, step=trainer.global_step, prefix="final")

        logger.info("\n✅ Training completed successfully!")

    except Exception as e:
        logger = logger or get_global_logger()
        logger.error(f"\n❌ Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    finally:
        gc.collect()
        torch.cuda.empty_cache()
