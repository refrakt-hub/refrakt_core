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
    opt_map = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg must convert to a dict, got {type(cfg_dict)}")
    optimizer_cfg = cfg_dict.get("optimizer")
    if not isinstance(optimizer_cfg, dict):
        raise TypeError(f"cfg.optimizer must be a dict, got {type(optimizer_cfg)}")

    if optimizer_cfg.get("generator") or optimizer_cfg.get("discriminator"):
        # Handle GAN-style optimizer without explicit 'components' key
        optimizer = {}
        for comp_name in ["generator", "discriminator"]:
            comp_cfg = optimizer_cfg.get(comp_name)
            if comp_cfg:
                if not isinstance(comp_cfg, dict):
                    raise TypeError(
                        f"{comp_name} config must be a dict, got {type(comp_cfg)}"
                    )
                opt_name = comp_cfg["name"]
                opt_cls = opt_map.get(opt_name.lower())
                if not opt_cls:
                    raise ValueError(
                        f"Unsupported optimizer for {comp_name}: {opt_name}"
                    )

                opt_params = OmegaConf.to_container(
                    comp_cfg.get("params", {}), resolve=True
                )
                if not isinstance(opt_params, dict) or not all(
                    isinstance(k, str) for k in opt_params.keys()
                ):
                    raise TypeError(
                        f"opt_params must be a dict with str keys, got {type(opt_params)} and keys {list(opt_params.keys()) if isinstance(opt_params, dict) else 'N/A'}"
                    )
                opt_params = typing.cast(Dict[str, Any], opt_params)

                # Get parameters for specific component
                if comp_name == "generator":
                    parameters = model.generator.parameters()
                elif comp_name == "discriminator":
                    parameters = model.discriminator.parameters()
                else:
                    raise ValueError(f"Unknown optimizer component: {comp_name}")

                optimizer[comp_name] = opt_cls(parameters, **opt_params)
                print(f"Optimizer ({comp_name}): {opt_name} with params: {opt_params}")

    elif optimizer_cfg.get("components"):
        # Handle multi-component optimizer (GAN)
        components = optimizer_cfg["components"]
        if not isinstance(components, dict):
            raise TypeError(f"components must be a dict, got {type(components)}")
        optimizer = {}
        for comp_name, comp_cfg in components.items():
            if not isinstance(comp_cfg, dict):
                raise TypeError(
                    f"component config must be a dict, got {type(comp_cfg)}"
                )
            opt_name = comp_cfg["name"]
            opt_cls = opt_map.get(opt_name.lower())
            if not opt_cls:
                raise ValueError(f"Unsupported optimizer for {comp_name}: {opt_name}")

            opt_params = OmegaConf.to_container(
                comp_cfg.get("params", {}), resolve=True
            )
            if not isinstance(opt_params, dict) or not all(
                isinstance(k, str) for k in opt_params.keys()
            ):
                raise TypeError(
                    f"opt_params must be a dict with str keys, got {type(opt_params)} and keys {list(opt_params.keys()) if isinstance(opt_params, dict) else 'N/A'}"
                )
            opt_params = typing.cast(Dict[str, Any], opt_params)

            # Get parameters for specific component
            if comp_name == "generator":
                parameters = model.generator.parameters()
            elif comp_name == "discriminator":
                parameters = model.discriminator.parameters()
            else:
                raise ValueError(f"Unknown optimizer component: {comp_name}")

            optimizer[comp_name] = opt_cls(parameters, **opt_params)
            print(f"Optimizer ({comp_name}): {opt_name} with params: {opt_params}")
    else:
        # Standard single optimizer (VAE, AE, etc.)
        opt_name = optimizer_cfg.get("name")
        if not isinstance(opt_name, str):
            raise TypeError(f"optimizer name must be a str, got {type(opt_name)}")
        opt_cls = opt_map.get(opt_name.lower())
        if not opt_cls:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        optimizer_params = optimizer_cfg.get("params", {}) or {}
        if OmegaConf.is_config(optimizer_params):
            optimizer_params = OmegaConf.to_container(optimizer_params, resolve=True)
        if not isinstance(optimizer_params, dict) or not all(
            isinstance(k, str) for k in optimizer_params.keys()
        ):
            raise TypeError(
                f"optimizer_params must be a dict with str keys, got {type(optimizer_params)} and keys {list(optimizer_params.keys()) if isinstance(optimizer_params, dict) else 'N/A'}"
            )
        optimizer_params = typing.cast(Dict[str, Any], optimizer_params)
        optimizer = opt_cls(model.parameters(), **optimizer_params)
        print(f"Optimizer: {opt_name} with params: {optimizer_params}")

    return optimizer
