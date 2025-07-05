"""
Loss builder for Refrakt.

This module provides utilities to construct and wrap loss functions from configuration dictionaries.
It supports single, multi-component, and GAN-style losses, and ensures all losses are wrapped for standardized output.

Typical usage involves passing a configuration (OmegaConf) and a modules registry to build loss functions for training.
"""

from omegaconf import OmegaConf
from torch import nn
from typing import Any, Dict, Union

from refrakt_core.api.builders.utils.loss_utils import (
    _validate_loss_config,
    _build_gan_style_loss,
    _build_multi_component_loss,
    _build_single_loss,
)

def build_loss(
    cfg: OmegaConf, modules: Dict[str, Any], device: str
) -> Union[nn.Module, Dict[str, nn.Module]]:
    """
    Build and wrap loss functions to return LossOutput objects.

    This function supports single losses, multi-component losses, and GAN-style (generator/discriminator) losses.
    All losses are wrapped to ensure a standardized output format compatible with the training pipeline.

    Args:
        cfg (OmegaConf): Configuration specifying the loss structure and parameters.
        modules (Dict[str, Any]): Registry of available loss functions and utilities.
        device (str): Device on which to place the loss modules.

    Returns:
        Union[nn.Module, Dict[str, nn.Module]]: A single loss module or a dictionary of loss modules (for multi-component or GAN losses).

    Raises:
        TypeError: If the configuration or its fields are not of the expected type.
        ValueError: If required loss components are missing or not found in the registry.
    """
    loss_cfg = _validate_loss_config(cfg)

    # GAN-style: generator/discriminator
    if loss_cfg.get("generator") or loss_cfg.get("discriminator"):
        return _build_gan_style_loss(loss_cfg, modules, device)

    # Multi-component losses
    elif loss_cfg.get("components"):
        return _build_multi_component_loss(loss_cfg, modules, device)

    # Single loss
    else:
        return _build_single_loss(loss_cfg, modules, device) 