"""
Utility functions for setting up and initializing datasets, dataloaders, and \
    model components.
"""

from typing import Any, Dict, Tuple, cast

import torch
from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.loss_builder import build_loss
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.builders.optimizer_builder import build_optimizer
from refrakt_core.api.builders.scheduler_builder import build_scheduler
from refrakt_core.api.core.components import ModelComponents


# pylint: disable=import-outside-toplevel
def import_modules() -> Dict[str, Any]:
    """
    Dynamically import and return registries and builder references.

    Returns:
        dict: A dictionary containing registry getter functions and \
            dataset/dataloader builders.
    """
    from refrakt_core.registry.loss_registry import get_loss
    from refrakt_core.registry.model_registry import get_model
    from refrakt_core.registry.trainer_registry import get_trainer

    return {
        "get_trainer": get_trainer,
        "get_loss": get_loss,
        "get_model": get_model,
        "build_dataset": build_dataset,
        "build_dataloader": build_dataloader,
    }


def setup_device() -> str:
    """
    Determine and return the appropriate device.

    Returns:
        str: 'cuda' if available, else 'cpu'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def build_datasets(cfg: OmegaConf) -> Tuple[Any, Any]:
    """
    Construct the training and validation datasets from configuration.

    Args:
        cfg (OmegaConf): Dataset configuration.

    Returns:
        Tuple: Training and validation dataset objects.
    """
    print("Building datasets...")
    train_dataset = build_dataset(cfg.dataset)  # type: ignore[attr-defined]

    # Modify config to set train=False for validation
    val_cfg = OmegaConf.merge(
        cfg.dataset, OmegaConf.create({"params": {"train": False}})  # type: ignore[attr-defined]
    )
    val_dataset = build_dataset(cast(DictConfig, val_cfg))

    return train_dataset, val_dataset


def build_dataloaders(
    train_dataset: Any, val_dataset: Any, cfg: OmegaConf
) -> Tuple[Any, Any]:
    """
    Build data loaders for training and validation.

    Args:
        train_dataset (Any): Training dataset.
        val_dataset (Any): Validation dataset.
        cfg (OmegaConf): Dataloader configuration.

    Returns:
        Tuple: Training and validation dataloaders.
    """
    print("Building data loaders...")
    train_loader = build_dataloader(train_dataset, cfg.dataloader)  # type: ignore[attr-defined]
    val_loader = build_dataloader(val_dataset, cfg.dataloader)  # type: ignore[attr-defined]
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


def build_model_components(cfg: OmegaConf) -> ModelComponents:
    """
    Assemble the model, loss function, optimizer, scheduler, and device into a \
        unified object.

    Args:
        cfg (OmegaConf): Configuration for all components.

    Returns:
        ModelComponents: Object containing all model-related components.
    """
    modules = import_modules()
    device = setup_device()

    model = build_model(cfg, modules, device)
    loss_fn = build_loss(cfg, modules, device)
    if isinstance(loss_fn, dict):
        loss_fn = next(iter(loss_fn.values()))
    optimizer = build_optimizer(cast(DictConfig, cfg), model)
    if isinstance(optimizer, dict):
        optimizer = optimizer["optimizer"]
    scheduler = build_scheduler(cfg, optimizer)

    return ModelComponents(model, loss_fn, optimizer, scheduler, device)


def flatten_and_filter_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested configuration dictionary and remove unsupported types.

    Args:
        cfg (dict): Nested configuration dictionary.

    Returns:
        dict: Flattened dictionary with only serializable types.
    """
    flat_cfg: Dict[str, Any] = {}

    def _flatten(prefix: str, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                flat_cfg[key] = v
            elif isinstance(v, dict):
                _flatten(key, v)
            # Skip other types (lists, None, etc.)

    _flatten("", cfg)
    return flat_cfg
