"""
Trainer module for Masked Siamese Networks (MSN).

This trainer handles the self-supervised training of MSN models using
random patch masking and EMA updates for target networks.
"""

from typing import Any, Callable, Dict, Optional

import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.utils.methods import random_patch_masking


@register_trainer("msn")
class MSNTrainer(BaseTrainer):
    """
    Trainer for Masked Siamese Networks (MSN).

    Args:
        model (Module): MSN model implementing the __call__ logic.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Optional validation data.
        loss_fn (Callable): Loss function taking (z_anchor, z_target, prototypes).
        optimizer_cls (Callable[..., Optimizer]): Optimizer constructor (e.g., Adam).
        optimizer_args (Optional[Dict[str, Any]]): Optimizer keyword arguments.
        device (str): Training device ("cuda" or "cpu").
        scheduler (Optional[Any]): Optional learning rate scheduler.
        **kwargs: Extra args forwarded to BaseTrainer.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer_cls: Callable[..., Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        scheduler: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, train_loader, val_loader, device, **kwargs)

        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.ema_base: float = kwargs.pop("ema_base", 0.996)
        self.grad_clip: Optional[float] = kwargs.pop("grad_clip", None)
        
        if optimizer_args is None:
            optimizer_args = {"lr": 1e-4}
        
        # Convert DictConfig to regular dict if needed
        from omegaconf import DictConfig
        if isinstance(optimizer_args, DictConfig):
            from omegaconf import OmegaConf
            optimizer_args = OmegaConf.to_container(optimizer_args, resolve=True)
        
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)
        self.global_step = 0

    def update_ema(self, momentum: float) -> None:
        """
        Update the exponential moving average (EMA) weights of the target networks.

        Args:
            momentum (float): The EMA momentum to use for update.
        """
        for param, ema_param in zip(
            self.model.encoder.parameters(),
            self.model.target_encoder.parameters(),
            strict=False,
        ):
            ema_param.data.mul_(momentum).add_((1 - momentum) * param.data)

        for param, ema_param in zip(
            self.model.projector.parameters(),
            self.model.target_projector.parameters(),
            strict=False,
        ):
            ema_param.data.mul_(momentum).add_((1 - momentum) * param.data)

    def train(self, num_epochs: int) -> None:
        """
        Train the MSN model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
        """
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

            for batch in pbar:
                x = batch[0].to(self.device)

                x_anchor = random_patch_masking(x, mask_ratio=0.6, patch_size=16)
                x_target = x

                self.optimizer.zero_grad()
                z_anchor, z_target, prototypes = self.model(x_anchor, x_target)
                loss = self.loss_fn(z_anchor, z_target, prototypes)

                loss.backward()
                if self.grad_clip is not None:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                momentum = self.ema_base + (1 - self.ema_base) * (
                    self.global_step / 10000
                )
                self.update_ema(momentum)
                self.global_step += 1

                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_loss = running_loss / len(self.train_loader)
            print(f"[Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}")

    def evaluate(self) -> float:
        """
        No-op evaluation for self-supervised training.

        Returns:
            float: Dummy 0.0 for pipeline compatibility.
        """
        print("[MSNTrainer] Evaluation not implemented for self-supervised pretraining.")
        return 0.0
