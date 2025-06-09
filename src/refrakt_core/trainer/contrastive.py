"""
Trainer module for contrastive learning models.

Implements the training and evaluation logic for models using contrastive objectives,
such as SimCLR or other self-supervised frameworks.
"""

from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer


@register_trainer("contrastive")
class ContrastiveTrainer(BaseTrainer):
    """
    Trainer class for contrastive learning tasks.

    This trainer handles training models using paired views (e.g., SimCLR-style) and 
    a contrastive loss function.

    Args:
        model (Module): The contrastive model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        loss_fn (Callable): Contrastive loss function (e.g., NT-Xent).
        optimizer_cls (Callable[..., Optimizer]): Optimizer class (e.g., torch.optim.Adam).
        optimizer_args (Optional[Dict[str, Any]]): Arguments to pass to the optimizer.
        device (str): Device string ("cuda" or "cpu").
        scheduler (Optional[Any]): Optional scheduler.
        **kwargs: Extra arguments forwarded to BaseTrainer.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer_cls: Callable[..., Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, train_loader, val_loader, device, **kwargs)
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.extra_params = kwargs

        if optimizer_args is None:
            optimizer_args = {"lr": 1e-3}

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)

    def train(self, num_epochs: int) -> None:
        """
        Train the model for a given number of epochs.

        Args:
            num_epochs (int): Number of training epochs.
        """
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0.0

            for batch in loop:
                view1, view2 = self._unpack_views(batch)
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)

                self.optimizer.zero_grad()
                z1 = self.model(view1)
                z2 = self.model(view2)

                loss = self.loss_fn(z1, z2)
                loss.backward()
                self.optimizer.step()

                loop.set_postfix({"loss": loss.item()})
                epoch_loss += loss.item()

            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch + 1}: learning rate = {current_lr:.6f}")

            current_accuracy = self.evaluate()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.save(suffix="best_model")
                print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")

            self.save(suffix="latest")
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1} Train Loss: {avg_epoch_loss:.4f}")

    def evaluate(self) -> float:
        """
        Evaluate the model using validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)
            for batch in loop:
                view1, view2 = self._unpack_views(batch)
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)

                z1 = self.model(view1)
                z2 = self.model(view2)

                loss = self.loss_fn(z1, z2)
                total_loss += loss.item()

                loop.set_postfix({"loss": loss.item()})

        avg_val_loss = total_loss / num_batches
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def _unpack_views(
        self, batch: Union[Dict[str, torch.Tensor], tuple, list]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unpack two views from the input batch.

        Args:
            batch (Union[Dict, list, tuple]): Input batch.

        Returns:
            tuple: Two tensors representing different augmentations of the same image.
        """
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch["view1"], batch["view2"]
        raise TypeError("Unsupported batch format for contrastive learning")
