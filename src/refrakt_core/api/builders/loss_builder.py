"""
Loss builder for Refrakt.

This module provides utilities to construct and wrap loss functions from configuration dictionaries.
It supports single, multi-component, and GAN-style losses, and ensures all losses are wrapped for standardized output.

Typical usage involves passing a configuration (OmegaConf) and a modules registry to build loss functions for training.
"""

from typing import Any, Dict, Union

from omegaconf import OmegaConf
from refrakt_core.registry.loss_registry import get_loss
from refrakt_core.wrappers.schema.default_loss import DefaultLossWrapper
from torch import nn


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

    def create_wrapped_loss(name: str, params: Dict[str, Any]) -> nn.Module:
        raw_loss = modules["get_loss"](name, **params)

        if isinstance(raw_loss, nn.Module) and hasattr(raw_loss, "forward"):
            return raw_loss.to(device)

        print(
            f"[DEBUG] Loss is not a full nn.Module, wrapping with DefaultLossWrapper: {type(raw_loss)}"
        )
        return DefaultLossWrapper(raw_loss).to(device)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg must convert to a dict, got {type(cfg_dict)}")
    loss_cfg = cfg_dict.get("loss")
    if not isinstance(loss_cfg, dict):
        raise TypeError(f"cfg.loss must be a dict, got {type(loss_cfg)}")

    # GAN-style: generator/discriminator
    if loss_cfg.get("generator") or loss_cfg.get("discriminator"):
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
                loss_fn[comp_name] = create_wrapped_loss(loss_name, loss_params)
                print(
                    f"[INFO] Loss ({comp_name}): {loss_name} with params: {loss_params}"
                )
        return loss_fn

    # Multi-component losses
    elif loss_cfg.get("components"):
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
            loss_fn[comp_name] = create_wrapped_loss(loss_name, loss_params)
            print(f"[INFO] Loss ({comp_name}): {loss_name} with params: {loss_params}")
        return loss_fn

    # Single loss
    else:
        loss_name = loss_cfg["name"]
        loss_params = loss_cfg.get("params", {})
        wrapped = create_wrapped_loss(loss_name, loss_params)
        print(f"[INFO] Loss: {loss_name} with params: {loss_params}")
        return wrapped
