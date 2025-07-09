"""
Optimizer utilities for Refrakt.

This module provides utility functions for optimizer building and configuration,
extracted from the main optimizer_builder to reduce complexity and improve \
    maintainability.

The module handles:
- Optimizer class mapping and registry
- Parameter validation and conversion
- Component-specific optimizer building (e.g., GAN generator/discriminator)
- Multi-component optimizer construction
- Standard single optimizer setup
- Model parameter extraction for different components

These utilities support various optimizer architectures including standard optimizers,
GAN-style optimizers with separate generator/discriminator optimizers, and
multi-component optimizers for complex models.

Typical usage involves passing configuration dictionaries and model objects
to build optimizers for different training scenarios.
"""

import typing
from typing import Any, Dict

import torch
from omegaconf import OmegaConf


def get_optimizer_map() -> Dict[str, Any]:
    """
    Get the mapping of optimizer names to PyTorch optimizer classes.

    This function provides a centralized registry of supported optimizers
    that can be used throughout the training pipeline.

    Returns:
        Dictionary mapping optimizer name strings to PyTorch optimizer classes
    """
    return {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }


def validate_optimizer_params(
    opt_params: Any, component_name: str = ""
) -> Dict[str, Any]:
    """
    Validate and convert optimizer parameters to the correct format.

    This function ensures that optimizer parameters are properly formatted
    as a dictionary with string keys, which is required for PyTorch optimizers.

    Args:
        opt_params: Optimizer parameters to validate and convert
        component_name: Optional component name for error reporting

    Returns:
        Validated optimizer parameters as a dictionary with string keys

    Raises:
        TypeError: If the parameters are not a dictionary or contain non-string keys
    """
    if not isinstance(opt_params, dict) or not all(
        isinstance(k, str) for k in opt_params.keys()
    ):
        raise TypeError(
            (
                "opt_params must be a dict with str keys, got "
                f"{type(opt_params)} "
                "and keys "
                f"{list(opt_params.keys()) if isinstance(opt_params, dict) else 'N/A'}"
            )
        )

    return typing.cast(Dict[str, Any], opt_params)


def get_model_parameters(model: Any, component_name: str) -> Any:
    """
    Get model parameters for a specific component.

    This function extracts the appropriate parameters from a model based on
    the component name. It's primarily used for GAN architectures where
    generator and discriminator have separate parameter sets.

    Args:
        model: Model object containing the component
        component_name: Name of the component ('generator' or 'discriminator')

    Returns:
        Parameter iterator for the specified component

    Raises:
        ValueError: If the component name is not recognized
    """
    if component_name == "generator":
        return model.generator.parameters()
    if component_name == "discriminator":
        return model.discriminator.parameters()
    raise ValueError(f"Unknown optimizer component: {component_name}")


def build_component_optimizer(
    comp_cfg: Dict[str, Any], model: Any, component_name: str, opt_map: Dict[str, Any]
) -> Any:
    """
    Build optimizer for a specific component of a model.

    This function constructs an optimizer for a specific component (e.g., generator
    or discriminator in GAN architectures) using the provided configuration.

    Args:
        comp_cfg: Component-specific optimizer configuration dictionary
        model: Model object containing the component
        component_name: Name of the component to build optimizer for
        opt_map: Dictionary mapping optimizer names to classes

    Returns:
        Configured optimizer instance for the specified component

    Raises:
        TypeError: If the component configuration is not a dictionary
        ValueError: If the optimizer name is not supported
    """
    if not isinstance(comp_cfg, dict):
        raise TypeError(f"{component_name} config must be a dict, got {type(comp_cfg)}")

    opt_name = comp_cfg["name"]
    opt_cls = opt_map.get(opt_name.lower())
    if not opt_cls:
        raise ValueError(f"Unsupported optimizer for {component_name}: {opt_name}")

    params = comp_cfg.get("params", {})
    if OmegaConf.is_config(params):
        opt_params = OmegaConf.to_container(params, resolve=True)
    else:
        opt_params = params
    opt_params = validate_optimizer_params(opt_params, component_name)

    parameters = get_model_parameters(model, component_name)
    optimizer = opt_cls(parameters, **opt_params)
    print(f"Optimizer ({component_name}): {opt_name} with params: {opt_params}")

    return optimizer


def build_gan_style_optimizer(
    optimizer_cfg: Dict[str, Any], model: Any, opt_map: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build GAN-style optimizer without explicit 'components' key.

    This function constructs separate optimizers for generator and discriminator
    components in GAN architectures. It supports optional components, allowing
    for cases where only one component is specified.

    Args:
        optimizer_cfg: Optimizer configuration dictionary containing \
            generator/discriminator settings
        model: Model object containing generator and discriminator components
        opt_map: Dictionary mapping optimizer names to classes

    Returns:
        Dictionary mapping component names ('generator', 'discriminator') to their
        respective optimizer instances

    Note:
        This function is designed for GAN architectures where generator and
        discriminator have separate optimizers but the configuration doesn't
        use an explicit 'components' key.
    """
    optimizer = {}
    for comp_name in ["generator", "discriminator"]:
        comp_cfg = optimizer_cfg.get(comp_name)
        if comp_cfg:
            optimizer[comp_name] = build_component_optimizer(
                comp_cfg, model, comp_name, opt_map
            )
    return optimizer


def build_multi_component_optimizer(
    optimizer_cfg: Dict[str, Any], model: Any, opt_map: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build multi-component optimizer with explicit 'components' key.

    This function constructs multiple optimizers for different model components
    using an explicit 'components' configuration structure.

    Args:
        optimizer_cfg: Optimizer configuration dictionary containing \
            component specifications
        model: Model object containing the components
        opt_map: Dictionary mapping optimizer names to classes

    Returns:
        Dictionary mapping component names to their respective optimizer instances

    Raises:
        TypeError: If the components configuration is not a dictionary
    """
    components = optimizer_cfg["components"]
    if not isinstance(components, dict):
        raise TypeError(f"components must be a dict, got {type(components)}")

    optimizer = {}
    for comp_name, comp_cfg in components.items():
        optimizer[comp_name] = build_component_optimizer(
            comp_cfg, model, comp_name, opt_map
        )
    return optimizer


def build_standard_optimizer(
    optimizer_cfg: Dict[str, Any], model: Any, opt_map: Dict[str, Any]
) -> Any:
    """
    Build standard single optimizer for models like VAE, AutoEncoder, etc.

    This function constructs a single optimizer for the entire model parameters.
    It's the simplest case where only one optimizer is needed for training.

    Args:
        optimizer_cfg: Optimizer configuration dictionary containing name and parameters
        model: Model object to create optimizer for
        opt_map: Dictionary mapping optimizer names to classes

    Returns:
        Configured optimizer instance for the entire model

    Raises:
        TypeError: If the optimizer name is not a string
        ValueError: If the optimizer name is not supported
    """
    opt_name = optimizer_cfg.get("name")
    if not isinstance(opt_name, str):
        raise TypeError(f"optimizer name must be a str, got {type(opt_name)}")

    opt_cls = opt_map.get(opt_name.lower())
    if not opt_cls:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    optimizer_params = optimizer_cfg.get("params", {}) or {}
    if OmegaConf.is_config(optimizer_params):
        optimizer_params = OmegaConf.to_container(optimizer_params, resolve=True)

    optimizer_params = validate_optimizer_params(optimizer_params)
    optimizer = opt_cls(model.parameters(), **optimizer_params)
    print(f"Optimizer: {opt_name} with params: {optimizer_params}")

    return optimizer
