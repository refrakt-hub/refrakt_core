"""Components module for bundling model, loss, optimizer, scheduler, and device."""

from typing import Optional

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class ModelComponents:
    """Container for model-related components."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda"
    ) -> None:
        self.model: nn.Module = model
        self.loss_fn: nn.Module = loss_fn
        self.optimizer: Optimizer = optimizer
        self.scheduler: Optional[_LRScheduler] = scheduler
        self.device: str = device
