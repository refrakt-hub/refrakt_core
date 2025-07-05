"""
Loss utilities for Refrakt.

This module contains internal helper functions used by the loss builder.
"""

from typing import Any, Dict, Union

from omegaconf import OmegaConf
from refrakt_core.registry.loss_registry import get_loss
from refrakt_core.wrappers.schema.default_loss import DefaultLossWrapper
from torch import nn


def _create_wrapped_loss(name: str, params: Dict[str, Any], modules: Dict[str, Any], device: str) -> nn.Module:
    """Create a wrapped loss function."""
    raw_loss = modules["get_loss"](name, **params)

    if isinstance(raw_loss, nn.Module) and hasattr(raw_loss, "forward"):
        return raw_loss.to(device)

    print(
        f"[DEBUG] Loss is not a full nn.Module, wrapping with DefaultLossWrapper: {type(raw_loss)}"
    )
    return DefaultLossWrapper(raw_loss).to(device)


def _validate_loss_config(cfg: OmegaConf) -> Dict[str, Any]:
    """Validate and extract loss configuration."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg must convert to a dict, got {type(cfg_dict)}")
    
    loss_cfg = cfg_dict.get("loss")
    if not isinstance(loss_cfg, dict):
        raise TypeError(f"cfg.loss must be a dict, got {type(loss_cfg)}")
    
    return loss_cfg


def _build_gan_style_loss(loss_cfg: Dict[str, Any], modules: Dict[str, Any], device: str) -> Dict[str, nn.Module]:
    """Build GAN-style generator/discriminator losses."""
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
            loss_fn[comp_name] = _create_wrapped_loss(loss_name, loss_params, modules, device)
            print(
                f"[INFO] Loss ({comp_name}): {loss_name} with params: {loss_params}"
            )
    return loss_fn


def _build_multi_component_loss(loss_cfg: Dict[str, Any], modules: Dict[str, Any], device: str) -> Dict[str, nn.Module]:
    """Build multi-component losses."""
    components = loss_cfg["components"]
    if not isinstance(components, dict):
        raise TypeError(f"components must be a dict, got {type(components)}")
    
    loss_fn = {}
    for comp_name, comp_cfg in components.items():
        if not isinstance(comp_cfg, dict):
            raise TypeError(
                f"component config must be a dict, got {type(comp_cfg)}"
            )
        loss_name = comp_cfg["name"]
        loss_params = comp_cfg.get("params", {})
        loss_fn[comp_name] = _create_wrapped_loss(loss_name, loss_params, modules, device)
        print(f"[INFO] Loss ({comp_name}): {loss_name} with params: {loss_params}")
    return loss_fn


def _build_single_loss(loss_cfg: Dict[str, Any], modules: Dict[str, Any], device: str) -> nn.Module:
    """Build single loss function."""
    loss_name = loss_cfg["name"]
    loss_params = loss_cfg.get("params", {})
    wrapped = _create_wrapped_loss(loss_name, loss_params, modules, device)
    print(f"[INFO] Loss: {loss_name} with params: {loss_params}")
    return wrapped