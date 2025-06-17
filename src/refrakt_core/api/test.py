"""The test module for Refrakt."""

import os
import sys
import traceback
from typing import Optional

import torch
from omegaconf import OmegaConf

from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.trainer_builder import initialize_trainer
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.logging import get_global_logger
from refrakt_core.api.utils.test_utils import _load_config, _build_test_loader, _load_model_checkpoint

from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.builders.loss_builder import build_loss
from refrakt_core.registry.model_registry import get_model
from refrakt_core.registry.loss_registry import get_loss
from refrakt_core.registry.trainer_registry import get_trainer
from refrakt_core.schema.artifact import ArtifactDumper
from refrakt_core.schema.model_output import ModelOutput

import warnings
warnings.filterwarnings("ignore")

def test(cfg, model_path=None, logger=None):
    try:
        # === Load config and logger ===
        config = _load_config(cfg)
        runtime_cfg = config.get("runtime", {})
        log_types = runtime_cfg.get("log_type", [])
        log_dir = runtime_cfg.get("log_dir", "./logs")
        mode = runtime_cfg.get("mode", "test")
        console = runtime_cfg.get("console", True)
        debug = runtime_cfg.get("debug", False)

        if logger is None:
            logger = RefraktLogger(
                model_name=config.model.name,
                log_dir=log_dir,
                log_types=log_types,
                console=console,
                debug=debug,
            )
        logger.log_config(OmegaConf.to_container(config, resolve=True))

        # === Set up modules and device ===
        modules = {
            "get_model": get_model,
            "get_loss": get_loss,
            "get_trainer": get_trainer
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # === Dataset, Model, Loss ===
        dataloader = _build_test_loader(config)
        model = build_model(config, modules=modules, device=device)
        loss_fn = build_loss(config, modules=modules, device=device)
        global_step = _load_model_checkpoint(model, model_path, device, logger)
        
        # === Artifact dumper ===
        artifact_log_every = config.get("artifacts", {}).get("log_every", 1)
        artifact_enabled = config.get("artifacts", {}).get("enabled", True)
        artifact_dumper = ArtifactDumper(
            enabled=artifact_enabled,
            base_path=os.path.join("./artifacts", mode.strip("/")),
            model_name=config.model.name,
            log_every=artifact_log_every, 
            logger=logger
        )

        # === Trainer ===
        trainer = initialize_trainer(
            cfg=config,
            model=model,
            train_loader=None,
            val_loader=dataloader,
            loss_fn=loss_fn,
            optimizer=None,
            scheduler=None,
            device=device,
            modules=modules,
            save_dir=None,
        )
        trainer.global_step = global_step
        trainer.logger = logger
        trainer.artifact_dumper = artifact_dumper
        

        # === Evaluation Phase ===
        logger.info("Running evaluation...")
        model.eval()
        eval_results = trainer.evaluate()
        
        if hasattr(logger, 'wandb') and hasattr(logger.wandb, 'step'):
            logger.wandb.step = 0

        # === Log model outputs per batch ===
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs = batch["input"] if isinstance(batch, dict) else batch[0]
                inputs = inputs.to(device)
                output = model(inputs)

                if isinstance(output, ModelOutput):
                    artifact_dumper.log_output(output, batch_id=i)
                else:
                    logger.warning(f"Model output at batch {i} is not a ModelOutput instance.")

        artifact_path = f"./artifacts/{mode}/{trainer.model.__class__.__name__}_outputs.pt"
        artifact_dumper.save(artifact_path)

        logger.info("✅ Evaluation completed.")
        return {
            "model": trainer.model,
            "evaluation_results": eval_results,
            "config": config,
            "artifacts_path": artifact_path,
        }

    except Exception as e:
        logger = logger or get_global_logger()
        logger.error(f"\n❌ Evaluation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return None
