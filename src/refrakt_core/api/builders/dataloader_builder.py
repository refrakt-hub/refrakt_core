"""
Dataloader builder for Refrakt.

This module provides a utility to construct PyTorch DataLoader objects from configuration dictionaries.
It ensures robust type checking and flexible parameter extraction for various dataset and dataloader settings.

Typical usage involves passing a dataset and a configuration (OmegaConf DictConfig) to build a DataLoader
with the appropriate batch size, shuffling, worker count, and other options.
"""

from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


def build_dataloader(dataset: Any, cfg: DictConfig) -> DataLoader[Any]:
    """
    Build a PyTorch DataLoader from a dataset and configuration.

    Args:
        dataset (Any): The dataset object to wrap in a DataLoader.
        cfg (DictConfig): OmegaConf configuration specifying DataLoader parameters.
            Expected keys include 'batch_size', 'shuffle', 'num_workers', and 'drop_last'.
            If 'params' is present, its contents are used as DataLoader arguments.

    Returns:
        DataLoader[Any]: Configured PyTorch DataLoader instance.

    Raises:
        TypeError: If the configuration cannot be converted to a dict or params is not a dict.
        KeyError: If 'batch_size' is missing from the configuration.
    """
    # Convert to native Python types
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Ensure cfg_dict is a dict
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg_dict must be a dict, got {type(cfg_dict)}")

    # Extract parameters
    params = cfg_dict.get("params", cfg_dict)
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dict, got {type(params)}")

    return DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=params.get("shuffle", False),
        num_workers=params.get("num_workers", 0),
        drop_last=params.get("drop_last", False),
    )
