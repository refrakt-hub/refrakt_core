"""
Optimizer builder for Refrakt.

This module provides utilities to construct optimizers from configuration dictionaries.
It supports standard, multi-component, and GAN-style optimizers, and ensures robust type
checking for all parameters.

The module handles:
- Optimizer configuration validation and parsing
- Standard optimizer instantiation (SGD, Adam, etc.)
- Multi-component optimizer construction for complex models
- GAN-style optimizer setup with generator/discriminator components
- Parameter type checking and validation
- Learning rate and weight decay configuration
- Model parameter filtering and grouping

Typical usage involves passing a configuration (OmegaConf DictConfig) and a \
    model to build optimizers for training.
"""

from typing import Any, Dict, Union

from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.builders.utils.optimizer_utils import (
    build_gan_style_optimizer,
    build_multi_component_optimizer,
    build_standard_optimizer,
    get_optimizer_map,
)


def build_optimizer(cfg: DictConfig, model: Any) -> Union[Any, Dict[str, Any]]:
    """
    Build optimizer(s) from configuration for a given model.

    This function supports standard optimizers, multi-component optimizers \
        (e.g., for GANs), and ensures all optimizer parameters are type-checked \
        and compatible with PyTorch.

    The function automatically detects the optimizer type based on configuration:
    - Standard: Single optimizer for the entire model
    - Multi-component: Multiple optimizers for different model components
    - GAN-style: Separate optimizers for generator and discriminator

    Args:
        cfg: Configuration object (DictConfig) specifying optimizer parameters,
             structure, and optional multi-component settings
        model: The model (or model components) whose parameters will be optimized.
               Can be a single model or a dictionary of model components

    Returns:
        Either a single optimizer or a dictionary of optimizers (Dict[str, Any])
        for multi-component or GAN-style setups. All optimizers are properly
        configured with the model parameters.

    Raises:
        TypeError: If the configuration or its fields are not of the expected type
        ValueError: If required optimizer components are missing or not found \
            in the registry
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
    if optimizer_cfg.get("components"):
        return build_multi_component_optimizer(optimizer_cfg, model, opt_map)

    # Standard single optimizer (VAE, AE, etc.)
    return build_standard_optimizer(optimizer_cfg, model, opt_map)
