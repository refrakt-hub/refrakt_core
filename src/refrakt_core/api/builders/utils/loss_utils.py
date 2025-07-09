"""
Loss utilities for Refrakt.

This module provides utility functions for loss function building and wrapping \
    operations, extracted from the main loss_builder to reduce complexity and \
    improve maintainability.

The module handles:
- Loss configuration validation and parsing
- Single loss function instantiation and wrapping
- Multi-component loss construction for complex scenarios
- GAN-style loss setup with generator/discriminator components
- Device placement and loss function optimization
- Standardized output format for training pipeline compatibility

These utilities support various loss architectures including single losses,
multi-component losses, and GAN-style losses with separate generator and
discriminator components.

Typical usage involves passing configuration dictionaries and module registries
to build loss functions for training pipelines.
"""

from typing import Any, Dict

from omegaconf import OmegaConf
from torch import nn

from refrakt_core.wrappers.schema.default_loss import DefaultLossWrapper


def _create_wrapped_loss(
    name: str, params: Dict[str, Any], modules: Dict[str, Any], device: str
) -> nn.Module:
    """
    Create a wrapped loss function with proper device placement.

    This function instantiates a loss function from the registry and ensures
    it's properly wrapped and moved to the target device. If the loss is not
    a full nn.Module, it wraps it with DefaultLossWrapper for consistency.

    Args:
        name: Name of the loss function to instantiate from the registry
        params: Loss-specific parameters dictionary
        modules: Registry dictionary containing available loss functions
        device: Target device string (e.g., "cuda", "cpu") for loss placement

    Returns:
        Loss function as nn.Module moved to the specified device

    Note:
        All losses are wrapped to ensure consistent interface and device placement,
        even if they're not full nn.Module instances.
    """
    raw_loss = modules["get_loss"](name, **params)

    if isinstance(raw_loss, nn.Module) and hasattr(raw_loss, "forward"):
        return raw_loss.to(device)

    print(
        f"[DEBUG] Loss is not a full nn.Module, wrapping with \
            DefaultLossWrapper: {type(raw_loss)}"
    )
    return DefaultLossWrapper(raw_loss).to(device)


def _validate_loss_config(cfg: OmegaConf) -> Dict[str, Any]:
    """
    Validate and extract loss configuration from the main configuration.

    This function validates the structure of the loss configuration and extracts
    the loss-specific settings. It ensures the configuration is properly formatted
    before loss building operations.

    Args:
        cfg: Configuration object (OmegaConf) containing loss specifications

    Returns:
        Dictionary containing validated loss configuration

    Raises:
        TypeError: If the configuration structure is invalid or required fields
                  are missing or of incorrect type.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg must convert to a dict, got {type(cfg_dict)}")

    loss_cfg = cfg_dict.get("loss")
    if not isinstance(loss_cfg, dict):
        raise TypeError(f"cfg.loss must be a dict, got {type(loss_cfg)}")

    return loss_cfg


def _build_gan_style_loss(
    loss_cfg: Dict[str, Any], modules: Dict[str, Any], device: str
) -> Dict[str, nn.Module]:
    """
    Build GAN-style generator/discriminator losses.

    This function constructs separate loss functions for generator and discriminator
    components in GAN architectures. It supports optional components, allowing
    for cases where only one component is specified.

    Args:
        loss_cfg: Loss configuration dictionary containing \
            generator/discriminator settings
        modules: Registry dictionary containing available loss functions
        device: Target device string for loss function placement

    Returns:
        Dictionary mapping component names ('generator', 'discriminator') to their
        respective loss functions as nn.Module instances

    Raises:
        TypeError: If component configurations are not properly formatted dictionaries
    """
    loss_fn = {}
    for comp_name in ["generator", "discriminator"]:
        comp_cfg = loss_cfg.get(comp_name)
        if comp_cfg:
            if not isinstance(comp_cfg, dict):
                raise TypeError(
                    f"{comp_name} config must be a dict, got {type(comp_cfg)}"
                )
            loss_name = comp_cfg["name"]
            loss_params = comp_cfg.get("params", {})
            loss_fn[comp_name] = _create_wrapped_loss(
                loss_name, loss_params, modules, device
            )
            print(f"[INFO] Loss ({comp_name}): {loss_name} with params: {loss_params}")
    return loss_fn


def _build_multi_component_loss(
    loss_cfg: Dict[str, Any], modules: Dict[str, Any], device: str
) -> Dict[str, nn.Module]:
    """
    Build multi-component losses for complex training scenarios.

    This function constructs multiple loss functions that can be combined or
    weighted during training. Each component is instantiated separately and
    can have its own parameters and configuration.

    Args:
        loss_cfg: Loss configuration dictionary containing component specifications
        modules: Registry dictionary containing available loss functions
        device: Target device string for loss function placement

    Returns:
        Dictionary mapping component names to their respective loss functions
        as nn.Module instances

    Raises:
        TypeError: If the components configuration is not a dictionary or if
                  individual component configurations are invalid
    """
    components = loss_cfg["components"]
    if not isinstance(components, dict):
        raise TypeError(f"components must be a dict, got {type(components)}")

    loss_fn = {}
    for comp_name, comp_cfg in components.items():
        if not isinstance(comp_cfg, dict):
            raise TypeError(f"component config must be a dict, got {type(comp_cfg)}")
        loss_name = comp_cfg["name"]
        loss_params = comp_cfg.get("params", {})
        loss_fn[comp_name] = _create_wrapped_loss(
            loss_name, loss_params, modules, device
        )
        print(f"[INFO] Loss ({comp_name}): {loss_name} with params: {loss_params}")
    return loss_fn


def _build_single_loss(
    loss_cfg: Dict[str, Any], modules: Dict[str, Any], device: str
) -> nn.Module:
    """
    Build a single loss function for standard training scenarios.

    This function constructs a single loss function from the configuration.
    It's the simplest case where only one loss function is needed for training.

    Args:
        loss_cfg: Loss configuration dictionary containing loss name and parameters
        modules: Registry dictionary containing available loss functions
        device: Target device string for loss function placement

    Returns:
        Loss function as nn.Module instance moved to the specified device

    Note:
        This is the default case for most training scenarios where a single
        loss function is sufficient for the training objective.
    """
    loss_name = loss_cfg["name"]
    loss_params = loss_cfg.get("params", {})
    wrapped = _create_wrapped_loss(loss_name, loss_params, modules, device)
    print(f"[INFO] Loss: {loss_name} with params: {loss_params}")
    return wrapped
