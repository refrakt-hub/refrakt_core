"""
Trainer module for supervised classification tasks.

Provides training and evaluation loops for standard supervised learning models.
"""

from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer


@register_trainer("supervised")
class SupervisedTrainer(BaseTrainer):
    """
    Supervised training loop for classification models.

    Args:
        model (Module): PyTorch model for supervised classification.
        train_loader (DataLoader): Training set loader.
        val_loader (DataLoader): Validation set loader.
        loss_fn (Callable): Loss function like CrossEntropyLoss.
        optimizer_cls (Callable[..., Optimizer]): Optimizer class.
        optimizer_args (Optional[Dict[str, Any]]): Optional optimizer keyword arguments.
        device (str): Device to train on ("cuda" or "cpu").
        scheduler (Optional[Any]): Optional learning rate scheduler.
        **kwargs: Additional arguments passed to BaseTrainer.
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
        
        # Fix: Properly handle optimizer_args as keyword arguments
        if optimizer_args is None:
            optimizer_args = {"lr": 1e-4}
        
        # Convert DictConfig to regular dict if needed
        from omegaconf import DictConfig
        if isinstance(optimizer_args, DictConfig):
            from omegaconf import OmegaConf
            optimizer_args = OmegaConf.to_container(optimizer_args, resolve=True)
        
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)
        self.scheduler = scheduler
        self.extra_params = kwargs

    def train(self, num_epochs: int) -> None:
        """
        Train the model for a given number of epochs.

        Args:
            num_epochs (int): Total number of training epochs.
        """
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch in loop:
                inputs, targets = self._unpack_batch(batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                loop.set_postfix({"loss": loss.item()})

            if self.scheduler:
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch + 1} complete. Learning rate: {lr:.6f}")

            current_accuracy = self.evaluate()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.save(suffix="best_model")
                print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")

            self.save(suffix="latest")

    def evaluate(self) -> float:
        """
        Evaluate model accuracy on validation set.

        Returns:
            float: Validation accuracy in [0, 1].
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)

            for batch in loop:
                inputs, targets = self._unpack_batch(batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                acc_str = f"{(correct / total * 100):.2f}%" if total > 0 else "0.00%"
                loop.set_postfix({"acc": acc_str})

        accuracy = correct / total if total > 0 else 0.0
        print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def _unpack_batch(
        self, batch: Union[tuple, list, Dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract input and target tensors from the batch.

        Args:
            batch (tuple/list/dict): Batch from DataLoader.

        Returns:
            Tuple of (input, target) tensors.
        """
        if isinstance(batch, (tuple, list)):
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch["input"], batch["target"]
        raise TypeError("Unsupported batch format")
