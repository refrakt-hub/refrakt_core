"""
Trainer module for DINO (self-distillation with no labels).

This trainer handles training of student-teacher models using momentum updates and
contrastive objectives, typically for self-supervised learning.
"""

from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import autocast
from torch.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer


@register_trainer("dino")
class DINOTrainer(BaseTrainer):
    """
    Trainer for DINO-style student-teacher models.

    Args:
        model (Module): DINO model with teacher-student mechanism.
        train_loader (DataLoader): Training data loader.
        val_loader (Optional[DataLoader]): Validation data loader.
        loss_fn (Callable): Loss function accepting (student_out, teacher_out).
        optimizer_cls (Optional[Callable]): Optimizer class, defaults to Adam.
        optimizer_args (Optional[Dict[str, Any]]): Args for optimizer.
        scheduler (Optional[Any]): Learning rate scheduler.
        device (str): Device to use ("cuda" or "cpu").
        **kwargs: Passed to BaseTrainer.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        optimizer_cls: Optional[Callable[..., Optimizer]] = None,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, train_loader, val_loader, device, **kwargs)

        if loss_fn is None:
            raise ValueError("loss_fn is required for DINOTrainer")
        self.loss_fn = loss_fn

        if optimizer_cls is None:
            optimizer_cls = torch.optim.Adam
        if optimizer_args is None:
            optimizer_args = {"lr": 1e-3}

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)
        self.scheduler = scheduler
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

    def _unpack_views(
        self, batch: Union[torch.Tensor, Dict[str, torch.Tensor], list, tuple]
    ) -> list[torch.Tensor]:
        """
        Unpack paired views from batch for contrastive training.

        Returns:
            List containing two tensors (view1, view2).
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            if all(isinstance(b, torch.Tensor) for b in batch):
                return [batch[0].to(self.device).float(), batch[1].to(self.device).float()]

        if isinstance(batch, torch.Tensor):
            if batch.ndim == 5 and batch.size(1) == 2:
                return [batch[:, 0].to(self.device).float(), batch[:, 1].to(self.device).float()]
            raise ValueError(f"Unexpected tensor batch shape: {batch.shape}")

        if isinstance(batch, dict):
            return [batch["view1"].to(self.device).float(), batch["view2"].to(self.device).float()]

        if isinstance(batch, (list, tuple)):
            view1_batch = []
            view2_batch = []
            for item in batch:
                if isinstance(item, (tuple, list)):
                    view1_batch.append(item[0])
                    view2_batch.append(item[1])
                elif isinstance(item, dict):
                    view1_batch.append(item["view1"])
                    view2_batch.append(item["view2"])
                else:
                    raise TypeError(f"Unexpected batch item type: {type(item)}")
            return [
                torch.stack(view1_batch).to(self.device).float(),
                torch.stack(view2_batch).to(self.device).float(),
            ]

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def train(self, num_epochs: int) -> None:
        """
        Train the DINO model for a given number of epochs.

        Args:
            num_epochs (int): Number of training epochs.
        """
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

            for batch in loop:
                try:
                    views = self._unpack_views(batch)

                    with autocast(device_type=self.device.type):
                        student_out = torch.stack(
                            [self.model(view, teacher=False) for view in views], dim=1
                        )
                        teacher_out = self.model(views[0], teacher=True).unsqueeze(1)
                        loss = self.loss_fn(student_out, teacher_out)

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.model.update_teacher()
                    total_loss += loss.item()
                    loop.set_postfix(loss=loss.item())

                except (RuntimeError, ValueError, TypeError) as e:
                    loop.write(f"[ERROR] Batch skipped due to error: {e}")

            if self.scheduler:
                self.scheduler.step()

            current_accuracy = self.evaluate()
            if current_accuracy and current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.save(suffix="best_model")
                print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")

            self.save(suffix="latest")
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self) -> Optional[float]:
        """
        Evaluate DINO model using validation set.

        Returns:
            Optional[float]: Average validation loss or None if no validation loader.
        """
        if self.val_loader is None:
            print("No validation loader provided")
            return None

        self.model.eval()
        total_loss = 0.0
        loop = tqdm(self.val_loader, desc="Evaluating", leave=True)

        with torch.no_grad():
            for batch in loop:
                try:
                    views = self._unpack_views(batch)
                    student_out = torch.stack(
                        [self.model(view, teacher=False) for view in views], dim=1
                    )
                    teacher_out = self.model(views[0], teacher=True).unsqueeze(1)
                    loss = self.loss_fn(student_out, teacher_out)
                    total_loss += loss.item()
                    loop.set_postfix(val_loss=loss.item())

                except (RuntimeError, ValueError, TypeError) as e:
                    loop.write(f"[ERROR] Validation batch skipped due to error: {e}")

        avg_val_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
