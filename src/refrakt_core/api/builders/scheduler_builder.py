"""
Scheduler builder for Refrakt.

This module provides utilities to construct learning rate schedulers from \
    configuration dictionaries. It supports a variety of PyTorch schedulers \
    and ensures robust type checking for all parameters.

The module handles:
- Scheduler configuration validation and parsing
- Learning rate scheduler instantiation (CosineAnnealingLR, StepLR, etc.)
- Parameter type checking and validation
- Optimizer integration and setup
- Supported scheduler types and their parameters
- Optional scheduler configuration (returns None if not specified)

Typical usage involves passing a configuration (OmegaConf) and an optimizer to \
    build a scheduler for training.
"""

from typing import Any, Optional

import torch
from omegaconf import OmegaConf


def build_scheduler(cfg: OmegaConf, optimizer: Any) -> Optional[Any]:
    """
    Build a learning rate scheduler from configuration for a given optimizer.

    This function supports several PyTorch learning rate schedulers and ensures all
    parameters are type-checked and compatible with the provided optimizer.

    Supported scheduler types:
    - cosine: CosineAnnealingLR for cosine annealing learning rate
    - steplr: StepLR for step-based learning rate decay
    - multisteplr: MultiStepLR for multi-step learning rate decay
    - exponential: ExponentialLR for exponential learning rate decay

    Args:
        cfg: Configuration object (OmegaConf) specifying the scheduler type,
             parameters, and optional settings
        optimizer: The optimizer to which the scheduler will be attached.
                  Can be any PyTorch optimizer or optimizer dictionary

    Returns:
        The instantiated scheduler object if configuration is provided,
        None if no scheduler is specified in the configuration

    Raises:
        TypeError: If the configuration or its fields are not of the expected type
        ValueError: If the specified scheduler type is not supported
    """
    scheduler = None
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg must convert to a dict, got {type(cfg_dict)}")
    scheduler_cfg = cfg_dict.get("scheduler")
    if scheduler_cfg and isinstance(scheduler_cfg, dict) and scheduler_cfg.get("name"):
        print("Building scheduler...")
        sched_map = {
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
            "steplr": torch.optim.lr_scheduler.StepLR,
            "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
            "exponential": torch.optim.lr_scheduler.ExponentialLR,
        }
        scheduler_name = scheduler_cfg.get("name")
        if not isinstance(scheduler_name, str):
            raise TypeError(f"scheduler name must be a str, got {type(scheduler_name)}")
        scheduler_cls = sched_map.get(scheduler_name.lower())
        if not scheduler_cls:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        scheduler_params = scheduler_cfg.get("params", {}) or {}
        if not isinstance(scheduler_params, dict):
            raise TypeError(
                f"scheduler_params must be a dict, got {type(scheduler_params)}"
            )
        scheduler = scheduler_cls(optimizer, **scheduler_params)
        print(f"Scheduler: {scheduler_name} with params: {scheduler_params}")

    return scheduler
