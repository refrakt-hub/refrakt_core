"""
Trainer module for autoencoder models.

This trainer is responsible for handling the training and validation logic
of autoencoder-based models using PyTorch.
"""

from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.schema.model_output import ModelOutput


@register_trainer("autoencoder")
class AETrainer(BaseTrainer):
    """
    Autoencoder Trainer.

    Handles training and evaluation loops for autoencoder models.

    Args:
        model (Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        loss_fn (Callable): Loss function.
        optimizer_cls (Callable[..., Optimizer]): Optimizer class (e.g., torch.optim.Adam).
        optimizer_args (Optional[Dict[str, Any]]): Arguments for optimizer instantiation.
        device (str): Device to use ("cuda" or "cpu").
        scheduler (Optional[Any]): Learning rate scheduler.
        **kwargs: Additional arguments forwarded to BaseTrainer.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable[[ModelOutput, torch.Tensor], torch.Tensor],
        optimizer_cls: Callable[..., Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        model_name = kwargs.pop("model_name", None)
        variant = kwargs.pop("model_variant", None)

        if model_name is not None:
            kwargs["model_name"] = model_name
        elif variant is not None:
            kwargs["model_name"] = f"autoencoder_{variant}"
        else:
            kwargs["model_name"] = "autoencoder_simple"

        super().__init__(model, train_loader, val_loader, device, **kwargs)
        
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        if optimizer_args is None:
            optimizer_args = {"lr": 1e-3}

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)

    def train(self, num_epochs: int) -> None:
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch in loop:
                inputs = self._extract_inputs(batch)
                inputs = inputs.to(self.device)

                self.optimizer.zero_grad()
                output: ModelOutput = self.model(inputs)
                loss = self.loss_fn(output, inputs)
                loss.backward()
                self.optimizer.step()

                loop.set_postfix({"loss": loss.item()})

            if self.scheduler:
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch + 1} complete. Learning rate: {lr:.6f}")

            current_loss = self.evaluate()
            if current_loss < best_loss:
                best_loss = current_loss
                self.save(suffix="best_model")
                print(f"New best model saved with loss: {best_loss:.4f}")

            self.save(suffix="latest")

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)
            for batch in loop:
                inputs = self._extract_inputs(batch)
                inputs = inputs.to(self.device)

                output: ModelOutput = self.model(inputs)
                loss = self.loss_fn(output, inputs)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
    
    def _extract_inputs(self, batch: Union[torch.Tensor, Dict, list, tuple]) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            return batch[0]
        if isinstance(batch, dict):
            return batch.get("image", next(iter(batch.values())))
        return batch