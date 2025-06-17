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

    try:
        cfg = OmegaConf.load(config_path)

        # === Logger Setup ===
        if logger is None:
            runtime_cfg = cfg.get("runtime", {})
            log_types = runtime_cfg.get("log_type", [])
            valid_log_types = {"wandb", "tensorboard"}
            unknown = set(log_types) - valid_log_types
            if unknown:
                raise ValueError(f"❌ Unsupported log_type(s): {unknown}. Supported: {valid_log_types}")

            logger = RefraktLogger(
                model_name=cfg.model.name,
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
        val_cfg = OmegaConf.merge(cfg.dataset, OmegaConf.create({"params": {"train": False}}))
        val_dataset = build_dataset(val_cfg)

        logger.info("Building data loaders...")
        train_loader = build_dataloader(train_dataset, cfg.dataloader)
        val_loader = build_dataloader(val_dataset, cfg.dataloader)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # === Model ===
        logger.info("Building model...")
        model = build_model(cfg, modules={"get_model": get_model}, device=device)

        # === Log Model Graph ===
        logger.info("Logging model graph...")
        try:
            # Get first batch
            example_batch = next(iter(train_loader))

            # Handle multiple batch formats
            if isinstance(example_batch, (tuple, list)):
                sample_input = example_batch[0]  # usually (inputs, targets)
            elif isinstance(example_batch, dict):
                sample_input = example_batch.get("input") or example_batch.get("image") or example_batch
            else:
                sample_input = example_batch

            # Move input to same device as model
            if isinstance(sample_input, torch.Tensor):
                sample_input = sample_input.to(next(model.parameters()).device)
            elif isinstance(sample_input, dict):
                sample_input = {k: v.to(next(model.parameters()).device) for k, v in sample_input.items()}

            # Forward once to get output
            with torch.no_grad():
                output = model(sample_input)

            # Call logger (uses .logits or .reconstruction or first tensor automatically)
            logger.log_model_graph(model, sample_input, model_output=output)

        except Exception as e:
            logger.warning(f"[RefraktLogger] Skipping model graph logging: {e}")

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
            logger.info(f"Scheduler: {cfg.scheduler.name} with params: {scheduler_params}")

        # === Trainer Setup ===
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

        # === Artifact Dumper ===
        artifact_log_every = cfg.get("artifacts", {}).get("log_every", 1)
        runtime_mode = cfg.get("runtime", {}).get("mode", "train")
        artifact_dumper = ArtifactDumper(
            enabled=True,
            model_name=cfg.model.name,
            base_path=os.path.join("./artifacts", runtime_mode.strip("/")),
            logger=logger
        )
        artifact_dumper.log_every = artifact_log_every
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
        final_metrics = trainer.train(num_epochs=num_epochs)

        logger.info("Saving model now...")
        trainer.save(path=model_path)

        config_save_path = os.path.join(
            trainer.save_dir or os.path.join("./artifacts", runtime_mode), 
            f"{trainer.model.__class__.__name__}.yaml"
        )
        OmegaConf.save(cfg, config_save_path)
        logger.info(f"Saved config to {config_save_path}")

        # Log final metrics
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