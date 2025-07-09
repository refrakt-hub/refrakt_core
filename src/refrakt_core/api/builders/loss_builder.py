"""
Loss builder for Refrakt.

This module provides utilities to construct and wrap loss functions from
configuration dictionaries. It supports single, multi-component, and GAN-style
losses, and ensures all losses are wrapped for standardized output.

The module handles:
- Loss configuration validation and parsing
- Single loss function instantiation and wrapping
- Multi-component loss construction for complex scenarios
- GAN-style loss setup with generator/discriminator components
- Device placement and loss function optimization
- Standardized output format for training pipeline compatibility

Typical usage involves passing a configuration (OmegaConf) and a modules
registry to build loss functions for training.
"""

from typing import Any, Dict, Union

from omegaconf import OmegaConf
from torch import nn

from refrakt_core.api.builders.utils.loss_utils import (
    _build_gan_style_loss,
    _build_multi_component_loss,
    _build_single_loss,
    _validate_loss_config,
)


def build_loss(
    cfg: OmegaConf, modules: Dict[str, Any], device: str
) -> Union[nn.Module, Dict[str, nn.Module]]:
    """
    Build and wrap loss functions to return LossOutput objects.

    This function supports single losses, multi-component losses, and GAN-style
    (generator/discriminator) losses. All losses are wrapped to ensure a
    standardized output format compatible with the training pipeline.

    The function automatically detects the loss type based on configuration:
    - Single loss: Direct loss function instantiation
    - Multi-component: Multiple loss functions combined
    - GAN-style: Separate generator and discriminator losses

    Args:
        cfg: Configuration object (OmegaConf) specifying the loss structure,
            parameters, and optional multi-component settings
        modules: Registry dictionary containing available loss functions and
            utilities
        device: Target device string (e.g., "cuda", "cpu") for loss function
            placement

    Returns:
        Either a single loss module (nn.Module) or a dictionary of loss modules
        (Dict[str, nn.Module]) for multi-component or GAN-style losses. All losses
        are wrapped to ensure standardized output format.

    Raises:
        TypeError: If the configuration or its fields are not of the expected type
        ValueError: If required loss components are missing or not found in the
            registry
    """
    loss_cfg = _validate_loss_config(cfg)

    # GAN-style: generator/discriminator
    if loss_cfg.get("generator") or loss_cfg.get("discriminator"):
        return _build_gan_style_loss(loss_cfg, modules, device)

    # Multi-component losses
    if loss_cfg.get("components"):
        return _build_multi_component_loss(loss_cfg, modules, device)

    # Single loss
    return _build_single_loss(loss_cfg, modules, device)
