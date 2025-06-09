"""# scheduler_builder.py"""
from typing import Any, Optional

import torch
from omegaconf import OmegaConf


def build_scheduler(cfg: OmegaConf, optimizer: Any) -> Optional[Any]:
    """Build learning rate scheduler"""
    scheduler = None
    if cfg.scheduler and cfg.scheduler.name:
        print("Building scheduler...")
        sched_map = {
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
            "steplr": torch.optim.lr_scheduler.StepLR,
            "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
            "exponential": torch.optim.lr_scheduler.ExponentialLR,
        }
        scheduler_cls = sched_map.get(cfg.scheduler.name.lower())
        if not scheduler_cls:
            raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")

        scheduler_params = cfg.scheduler.params or {}
        scheduler = scheduler_cls(optimizer, **scheduler_params)
        print(f"Scheduler: {cfg.scheduler.name} with params: {scheduler_params}")

    return scheduler
