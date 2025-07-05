"""
Optimizer builder for Refrakt.

This module provides utilities to construct optimizers from configuration dictionaries.
It supports standard, multi-component, and GAN-style optimizers, and ensures robust type checking for all parameters.

Typical usage involves passing a configuration (OmegaConf DictConfig) and a model to build optimizers for training.
"""

import typing
from typing import Any, Dict, Union

import torch
from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.builders.utils.optimizer_utils import (
    get_optimizer_map,
    validate_optimizer_params,
    get_model_parameters,
    build_component_optimizer,
    build_gan_style_optimizer,
    build_multi_component_optimizer,
    build_standard_optimizer
)


def build_optimizer(cfg: DictConfig, model: Any) -> Union[Any, Dict[str, Any]]:
    """
    Build optimizer(s) from configuration for a given model.

    This function supports standard optimizers, multi-component optimizers (e.g., for GANs),
    and ensures all optimizer parameters are type-checked and compatible with PyTorch.

    Args:
        cfg (DictConfig): OmegaConf configuration specifying optimizer parameters and structure.
        model (Any): The model (or model components) whose parameters will be optimized.

    Returns:
        Union[Any, Dict[str, Any]]: A single optimizer or a dictionary of optimizers (for multi-component setups).

    Raises:
        TypeError: If the configuration or its fields are not of the expected type.
        ValueError: If required optimizer components are missing or not found in the registry.
    """
    print("Building optimizer...")
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg must convert to a dict, got {type(cfg_dict)}")
    
    optimizer_cfg = cfg_dict.get("optimizer")
    if not isinstance(optimizer_cfg, dict):
        raise TypeError(f"cfg.optimizer must be a dict, got {type(optimizer_cfg)}")

    opt_map = get_optimizer_map()

    # Handle GAN-style optimizer without explicit 'components' key
    if optimizer_cfg.get("generator") or optimizer_cfg.get("discriminator"):
        return build_gan_style_optimizer(optimizer_cfg, model, opt_map)
    
    # Handle multi-component optimizer (GAN)
    elif optimizer_cfg.get("components"):
        return build_multi_component_optimizer(optimizer_cfg, model, opt_map)
    
    # Standard single optimizer (VAE, AE, etc.)
    else:
        return build_standard_optimizer(optimizer_cfg, model, opt_map)
