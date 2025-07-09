"""
ContrastiveTrainer implementation for contrastive learning tasks.

This module defines the ContrastiveTrainer class, which handles training and evaluation
of models using contrastive objectives (e.g., SimCLR, MoCo, etc.).
It supports logging, artifact dumping, and checkpointing.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.amp.grad_scaler import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.trainer.utils.contrastive_utils import (
    handle_contrastive_evaluation_step,
    handle_contrastive_training_step,
)
from refrakt_core.utils.methods import unpack_views_from_batch


@register_trainer("contrastive")
class ContrastiveTrainer(BaseTrainer):
    """
    Trainer for contrastive learning tasks (e.g., SimCLR, MoCo).

    Handles training, evaluation, logging, and artifact dumping for contrastive models.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader[Any],
        val_loader: Optional[DataLoader[Any]] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], LossOutput]] = None,
        optimizer_cls: Optional[Callable[..., Optimizer]] = None,
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        artifact_dumper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ContrastiveTrainer.

        Args:
            model (Module): The model to be trained.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader], optional): DataLoader for validation data.
            loss_fn (Callable, optional): Loss function for contrastive learning.
            optimizer_cls (Callable[..., Optimizer], optional): Optimizer class.
            optimizer_args (Optional[Dict[str, Any]], optional): Arguments for the optimizer.
            device (str, optional): Device to use (default: "cuda").
            scheduler (Optional[Any], optional): Learning rate scheduler.
            artifact_dumper (Optional[Any], optional): Artifact logger/dumper.
            **kwargs: Additional keyword arguments.
        """
        if val_loader is None:
            val_loader = DataLoader(TensorDataset())
        super().__init__(
            model,
            train_loader,
            val_loader,
            device,
            artifact_dumper=artifact_dumper,
            **kwargs,
        )

        if loss_fn is None:
            raise ValueError("loss_fn is required for ContrastiveTrainer")
        self.loss_fn = loss_fn

        if optimizer_cls is None:
            optimizer_cls = torch.optim.Adam
        if optimizer_args is None:
            optimizer_args = {"lr": 1e-3}

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)
        self.scheduler = scheduler
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        self.global_step = 0
        self.grad_log_interval = kwargs.get("grad_log_interval", 100)
        self.param_log_interval = kwargs.get("param_log_interval", 500)
        self.log_every = (
            getattr(self.artifact_dumper, "log_every", 10)
            if self.artifact_dumper
            else None
        )

    def _unpack_views(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unpack two augmented views from a batch for contrastive learning.

        Args:
            batch (Any): Batch from DataLoader.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two augmented views.
        """
        views = unpack_views_from_batch(batch, str(self.device))
        if isinstance(views, list):
            if len(views) != 2:
                raise ValueError("Expected exactly two views for contrastive learning.")
            return (views[0], views[1])
        raise ValueError("Expected exactly two views for contrastive learning.")

    def _get_logger(self) -> Optional[Any]:
        """
        Retrieve the logger from the artifact dumper if available.

        Returns:
            Optional[Any]: Logger object if available, else None.
        """
        return getattr(self.artifact_dumper, "logger", None)

    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
        """
        best_loss = float("inf")
        avg_loss = 0.0
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            loop = tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True
            )
            for batch_id, batch in enumerate(loop):
                try:
                    view1, view2 = self._unpack_views(batch)
                    batch_data = (view1, view2)

                    loss_value, success = handle_contrastive_training_step(
                        model=self.model,
                        batch=batch_data,
                        device=self.device,
                        loss_fn=self.loss_fn,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                        global_step=self.global_step,
                        artifact_dumper=self.artifact_dumper,
                        batch_id=batch_id,
                        grad_log_interval=self.grad_log_interval,
                        param_log_interval=self.param_log_interval,
                        logger=self._get_logger(),
                    )

                    if success:
                        total_loss += loss_value
                        loop.set_postfix(loss=loss_value)
                        self.global_step += 1
                    else:
                        loop.write("[WARNING] Skipping batch due to None outputs")

                except (RuntimeError, ValueError, TypeError) as e:
                    loop.write(f"[ERROR] Batch skipped due to error: {e}")

            if self.scheduler and not isinstance(self.scheduler, dict):
                self.scheduler.step()
            current_loss = self.evaluate()
            if current_loss is not None and current_loss < best_loss:
                best_loss = current_loss
                self.save(suffix="best_model")
                print(f"New best model saved with loss: {best_loss:.4f}")
            self.save(suffix="latest")
            avg_loss = (
                total_loss / len(self.train_loader)
                if len(self.train_loader) > 0
                else 0.0
            )
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        return {"final_loss": avg_loss, "best_loss": best_loss}

    def evaluate(self) -> Optional[float]:
        """
        Evaluate the model on the validation set.

        Returns:
            Optional[float]: Average validation loss, or None if no validation loader.
        """
        if self.val_loader is None:
            pass

        self.model.eval()
        total_loss = 0.0
        loop = tqdm(self.val_loader, desc="Evaluating", leave=True)

        with torch.no_grad():
            for batch_id, batch in enumerate(loop):
                try:
                    view1, view2 = self._unpack_views(batch)
                    batch_data = (view1, view2)

                    loss_value, success = handle_contrastive_evaluation_step(
                        model=self.model,
                        batch=batch_data,
                        device=self.device,
                        loss_fn=self.loss_fn,
                        global_step=self.global_step,
                        artifact_dumper=self.artifact_dumper,
                        batch_id=batch_id,
                    )

                    if success:
                        total_loss += loss_value
                        loop.set_postfix(val_loss=loss_value)
                    else:
                        continue

                except (RuntimeError, ValueError, TypeError) as e:
                    loop.write(f"[ERROR] Validation batch skipped due to error: {e}")

        avg_val_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
