"""
Optimizer builder utilities.

This module contains utility functions for building optimizers.
"""

import typing
from typing import Any, Dict

import torch
from omegaconf import OmegaConf


def get_optimizer_map() -> Dict[str, Any]:
    """Get the mapping of optimizer names to classes."""
    return {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }


def validate_optimizer_params(opt_params: Any, component_name: str = "") -> Dict[str, Any]:
    """Validate and convert optimizer parameters."""
    if not isinstance(opt_params, dict) or not all(
        isinstance(k, str) for k in opt_params.keys()
    ):
        raise TypeError(
            f"opt_params must be a dict with str keys, got {type(opt_params)} and keys {list(opt_params.keys()) if isinstance(opt_params, dict) else 'N/A'}"
        )
    return typing.cast(Dict[str, Any], opt_params)


def get_model_parameters(model: Any, component_name: str) -> Any:
    """Get model parameters for a specific component."""
    if component_name == "generator":
        return model.generator.parameters()
    elif component_name == "discriminator":
        return model.discriminator.parameters()
    else:
        raise ValueError(f"Unknown optimizer component: {component_name}")


def build_component_optimizer(comp_cfg: Dict[str, Any], model: Any, component_name: str, opt_map: Dict[str, Any]) -> Any:
    """Build optimizer for a specific component."""
    if not isinstance(comp_cfg, dict):
        raise TypeError(f"{component_name} config must be a dict, got {type(comp_cfg)}")
    
    opt_name = comp_cfg["name"]
    opt_cls = opt_map.get(opt_name.lower())
    if not opt_cls:
        raise ValueError(f"Unsupported optimizer for {component_name}: {opt_name}")

    opt_params = OmegaConf.to_container(comp_cfg.get("params", {}), resolve=True)
    opt_params = validate_optimizer_params(opt_params, component_name)
    
    parameters = get_model_parameters(model, component_name)
    optimizer = opt_cls(parameters, **opt_params)
    print(f"Optimizer ({component_name}): {opt_name} with params: {opt_params}")
    
    return optimizer


def build_gan_style_optimizer(optimizer_cfg: Dict[str, Any], model: Any, opt_map: Dict[str, Any]) -> Dict[str, Any]:
    """Build GAN-style optimizer without explicit 'components' key."""
    optimizer = {}
    for comp_name in ["generator", "discriminator"]:
        comp_cfg = optimizer_cfg.get(comp_name)
        if comp_cfg:
            optimizer[comp_name] = build_component_optimizer(comp_cfg, model, comp_name, opt_map)
    return optimizer


def build_multi_component_optimizer(optimizer_cfg: Dict[str, Any], model: Any, opt_map: Dict[str, Any]) -> Dict[str, Any]:
    """Build multi-component optimizer (GAN with explicit components)."""
    components = optimizer_cfg["components"]
    if not isinstance(components, dict):
        raise TypeError(f"components must be a dict, got {type(components)}")
    
    optimizer = {}
    for comp_name, comp_cfg in components.items():
        optimizer[comp_name] = build_component_optimizer(comp_cfg, model, comp_name, opt_map)
    return optimizer


def build_standard_optimizer(optimizer_cfg: Dict[str, Any], model: Any, opt_map: Dict[str, Any]) -> Any:
    """Build standard single optimizer (VAE, AE, etc.)."""
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