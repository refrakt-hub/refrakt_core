"""The test module for Refrakt."""

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


def test(cfg, model_path=None, logger=None):
    # === Load config and logger ===
    config = _load_config(cfg)
    logger = logger or get_global_logger()
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # === Set up modules and device ===
    modules = {"get_model": get_model, 
               "get_loss": get_loss, 
               "get_trainer": get_trainer}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Dataset & Model ===
    dataloader = _build_test_loader(config)
    model = build_model(config, modules=modules, device=device)
    loss_fn = build_loss(config, modules=modules, device=device)
    _load_model_checkpoint(model, model_path, device, logger)

    # === Artifact dumper ===
    artifact_dumper = ArtifactDumper(enabled=True, base_path="./artifacts/test")

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
    trainer.logger = logger

    # Attach artifact dumper to trainer if supported
    if hasattr(trainer, "artifact_dumper"):
        trainer.artifact_dumper = artifact_dumper

    # === Evaluate and log outputs ===
    logger.info("Running evaluation...")
    model.eval()
    eval_results = trainer.evaluate()

    # Dump model outputs for posthoc visualization
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch["input"] if isinstance(batch, dict) else batch[0]
            inputs = inputs.to(device)
            output = model(inputs)
            if isinstance(output, ModelOutput):
                artifact_dumper.log_output(output, batch_id=i)

    artifact_dumper.save(f"{trainer.model.__class__.__name__}_outputs.pt")

    logger.info("Evaluation completed.")
    return {
        "model": trainer.model,
        "evaluation_results": eval_results,
        "config": config,
        "artifacts_path": f"./artifacts/test/{trainer.model.__class__.__name__}_outputs.pt"
    }
