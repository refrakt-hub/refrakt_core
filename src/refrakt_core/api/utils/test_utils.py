"""
Utility functions for testing in Refrakt.

Includes config loading, test dataloader building, and model checkpoint loading helpers.
"""

import glob
import os
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.utils.methods import extract_visual_tensor


def _load_config(cfg: Any) -> Any:
    """
    Load an OmegaConf config from a file path or return as-is if already a config.

    Args:
        cfg (Any): Path to config file or OmegaConf config.

    Returns:
        Any: Loaded configuration object (DictConfig or ListConfig).
    """
    return OmegaConf.load(cfg) if isinstance(cfg, str) else cfg


def _build_test_loader(config: Any) -> Any:
    """
    Build a test dataloader from the given config.

    Args:
        config (Any): Configuration object (DictConfig or ListConfig).

    Returns:
        Any: PyTorch DataLoader for test data.
    """
    test_cfg = OmegaConf.merge(
        config.dataset, OmegaConf.create({"params": {"train": False}})
    )
    # Ensure test_cfg is a DictConfig
    from omegaconf import ListConfig

    if isinstance(test_cfg, ListConfig):
        test_cfg = OmegaConf.create(OmegaConf.to_container(test_cfg, resolve=True))
    if not isinstance(test_cfg, DictConfig):
        raise TypeError("test_cfg must be a DictConfig after conversion.")
    dataset = build_dataset(test_cfg)
    return build_dataloader(dataset, config.dataloader)


def _load_model_checkpoint(
    model: torch.nn.Module,
    model_path: Optional[str],
    device: torch.device,
    logger: Any,
) -> int:
    """
    Load a model checkpoint, with fallback logic for missing files and variants.

    Args:
        model (torch.nn.Module): The model to load weights into.
        model_path (Optional[str]): Path to the checkpoint file.
        device (torch.device): Device to map the checkpoint to.
        logger (Any): Logger for warnings and info.

    Returns:
        int: The global step from the checkpoint, or 0 if not found.

    Raises:
        FileNotFoundError: If no valid checkpoint is found.
    """
    import typing
    from collections import defaultdict

    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode
    from torch.serialization import add_safe_globals

    # Allow OmegaConf configs to be unpickled safely
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

    if model_path is None:
        logger.warning("No model checkpoint provided — using random init weights")
        return 0

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.info(f"Loaded model from {model_path}")
        return checkpoint.get("global_step", 0)

    # If file doesn't exist, try fallback logic
    base_dir = os.path.dirname(model_path)
    base_name = os.path.splitext(os.path.basename(model_path))[0]  # autoencoder_simple

    # Try exact match first
    exact_match = os.path.join(base_dir, f"{base_name}.pth")
    if os.path.exists(exact_match):
        checkpoint = torch.load(exact_match, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.warning(f"⚠️ Falling back to exact checkpoint: {exact_match}")
        return checkpoint.get("global_step", 0)

    # Try matching variants like _latest, _final
    pattern = os.path.join(base_dir, f"{base_name}_*.pth")
    candidates = glob.glob(pattern)
    if candidates:
        # Prefer latest or final first
        preferred = [c for c in candidates if "latest" in c or "final" in c]
        fallback_path = max(preferred or candidates, key=os.path.getmtime)

        checkpoint = torch.load(fallback_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.warning(f"⚠️ Falling back to available checkpoint: {fallback_path}")
        return checkpoint.get("global_step", 0)

    logger.error(f"Model path does not exist: {model_path}")
    raise FileNotFoundError(model_path)
