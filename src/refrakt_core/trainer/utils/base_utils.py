"""
Utility functions for base trainers.
"""

import os
from typing import Any, Dict, Optional

import torch


def setup_safe_globals() -> None:
    """
    Setup safe globals for torch.load to handle omegaconf objects.
    """
    import typing
    from collections import defaultdict

    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode
    from torch.serialization import add_safe_globals

    add_safe_globals(
        [
            ListConfig,
            DictConfig,
            ContainerMetadata,
            typing.Any,
            list,
            dict,
            defaultdict,
            int,
            float,
            AnyNode,
            Metadata,
        ]
    )


def load_checkpoint(path: str, device: str) -> Any:
    """
    Load checkpoint from a specific path.

    Args:
        path: Path to checkpoint file
        device: Target device

    Returns:
        Checkpoint dictionary

    Raises:
        OSError: If loading fails due to file issues
        RuntimeError: If loading fails due to state issues
    """
    setup_safe_globals()
    return torch.load(path, map_location=device, weights_only=False)


def determine_checkpoint_path(
    path: Optional[str],
    suffix: str,
    save_dir: str,
    model_name: str,
) -> str:
    """
    Determine the checkpoint path to load from.

    Args:
        path: Custom file path (if provided)
        suffix: Suffix for fallback path
        save_dir: Directory to save checkpoints
        model_name: Name of the model

    Returns:
        Path to load checkpoint from
    """
    if path is not None:
        return path

    if suffix == "best_model":
        return os.path.join(save_dir, f"{model_name}.pth")
    else:
        base_path = os.path.join(save_dir, f"{model_name}.pth")
        fallback_path = os.path.join(save_dir, f"{model_name}_{suffix}.pth")

        if os.path.exists(base_path):
            print(f"[INFO] Loading base model from: {base_path}")
            return base_path
        else:
            print(f"[INFO] Base model not found, falling back to: {fallback_path}")
            return fallback_path


def load_optimizer_state(
    optimizer: Any,
    optimizer_state: Dict[str, Any],
) -> None:
    """
    Load optimizer state from checkpoint.

    Args:
        optimizer: Optimizer to load state into
        optimizer_state: Optimizer state from checkpoint
    """
    if optimizer is None:
        return

    if isinstance(optimizer, dict):
        for k, opt in optimizer.items():
            if k in optimizer_state:
                opt.load_state_dict(optimizer_state[k])
    else:
        optimizer.load_state_dict(optimizer_state)


def load_scheduler_state(
    scheduler: Any,
    scheduler_state: Dict[str, Any],
) -> None:
    """
    Load scheduler state from checkpoint.

    Args:
        scheduler: Scheduler to load state into
        scheduler_state: Scheduler state from checkpoint
    """
    if scheduler is None:
        return

    if isinstance(scheduler, dict):
        for k, sch in scheduler.items():
            if k in scheduler_state:
                sch.load_state_dict(scheduler_state[k])
    else:
        scheduler.load_state_dict(scheduler_state)
