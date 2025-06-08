# [file content begin]
from typing import Any, Tuple

import torch
from omegaconf import OmegaConf

from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.loss_builder import build_loss
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.builders.optimizer_builder import build_optimizer
from refrakt_core.api.builders.scheduler_builder import build_scheduler
from refrakt_core.api.core.components import ModelComponents


# pylint: disable=import-outside-toplevel
def import_modules():
    """Import necessary modules"""
    from refrakt_core.registry.loss_registry import get_loss
    from refrakt_core.registry.model_registry import get_model
    from refrakt_core.registry.trainer_registry import get_trainer

    return {
        "get_trainer": get_trainer,
        "get_loss": get_loss,
        "get_model": get_model,
        "build_dataset": build_dataset,
        "build_dataloader": build_dataloaders,
    }


def setup_device() -> str:
    """Setup and return the appropriate device"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def build_datasets(cfg: OmegaConf) -> Tuple[Any, Any]:
    """Build train and validation datasets"""
    print("Building datasets...")
    train_dataset = build_dataset(cfg.dataset)

    # For validation, use same dataset config but with train=False
    val_cfg = OmegaConf.merge(
        cfg.dataset, OmegaConf.create({"params": {"train": False}})
    )
    val_dataset = build_dataset(val_cfg)

    return train_dataset, val_dataset


def build_dataloaders(
    train_dataset: Any, val_dataset: Any, cfg: OmegaConf
) -> Tuple[Any, Any]:
    """Build train and validation dataloaders"""
    print("Building data loaders...")
    train_loader = build_dataloader(train_dataset, cfg.dataloader)
    val_loader = build_dataloader(val_dataset, cfg.dataloader)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


def build_model_components(cfg: OmegaConf) -> ModelComponents:
    """Build all model-related components"""
    modules = import_modules()
    device = setup_device()
    # Build model
    model = build_model(cfg, modules, device)
    # Build loss function
    loss_fn = build_loss(cfg, modules, device)
    # Build optimizer
    optimizer = build_optimizer(cfg, model)
    # Build scheduler
    scheduler = build_scheduler(cfg, optimizer)
    return ModelComponents(model, loss_fn, optimizer, scheduler, device)


def flatten_and_filter_config(cfg: dict) -> dict:
    flat_cfg = {}

    def _flatten(prefix, d):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                flat_cfg[key] = v
            elif isinstance(v, dict):
                _flatten(key, v)
            # skip others (lists, None, etc.)

    _flatten("", cfg)
    return flat_cfg
