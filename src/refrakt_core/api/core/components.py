"""
Components module for bundling model, loss, optimizer, scheduler, and device.
"""

from typing import Optional

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class ModelComponents:
    """
    Container for model-related components including model, loss function,
    optimizer, scheduler, and device.

    Attributes:
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer for training.
        scheduler (Optional[_LRScheduler]): The learning rate scheduler (optional).
        device (str): The device on which the model is run (e.g., 'cuda' or 'cpu').
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
    ) -> None:
        """
        Initialize ModelComponents with all required subcomponents.

        Args:
            model (nn.Module): The neural network model.
            loss_fn (nn.Module): The loss function.
            optimizer (Optimizer): The optimizer for training.
            scheduler (Optional[_LRScheduler], optional): The learning rate scheduler. \
                Defaults to None.
            device (str, optional): The device to use. Defaults to 'cuda'.
        """
        self.model: nn.Module = model
        self.loss_fn: nn.Module = loss_fn
        self.optimizer: Optimizer = optimizer
        self.scheduler: Optional[_LRScheduler] = scheduler
        self.device: str = device
