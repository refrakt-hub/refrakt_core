"""
MSNTrainer implementation for Masked Siamese Network (MSN) training tasks.

This module defines the MSNTrainer class, which handles training and evaluation
of MSN models for self-supervised learning. It supports logging, artifact dumping,
and checkpointing.
"""

from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.trainer.utils.msn_utils import (
    handle_msn_training_step,
    prepare_msn_inputs,
)


@register_trainer("msn")
class MSNTrainer(BaseTrainer):
    """
    Trainer for Masked Siamese Network (MSN) self-supervised learning tasks.

    Handles training, evaluation, logging, and artifact dumping for MSN models.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        loss_fn: Callable[..., Any],
        optimizer_cls: Callable[..., Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        artifact_dumper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the MSNTrainer.

        Args:
            model (Module): The MSN model to be trained.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            loss_fn (Callable): Loss function for MSN training.
            optimizer_cls (Callable[..., Optimizer]): Optimizer class.
            optimizer_args (Optional[Dict[str, Any]], optional): Arguments for the optimizer.
            device (str, optional): Device to use (default: "cuda").
            scheduler (Optional[Any], optional): Learning rate scheduler.
            artifact_dumper (Optional[Any], optional): Artifact logger/dumper.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            model,
            train_loader,
            val_loader,
            device,
            artifact_dumper=artifact_dumper,
            **kwargs,
        )
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.artifact_dumper = artifact_dumper
        self.extra_params = kwargs

        self.optimizer: Optional[Union[Optimizer, Dict[str, Optimizer]]] = (
            optimizer_cls(self.model.parameters(), **(optimizer_args or {"lr": 1e-3}))
        )

        # Logging setup
        self.log_every = (
            getattr(artifact_dumper, "log_every", 10) if artifact_dumper else None
        )
        self.global_step = 0

    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        Train the MSN model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
        """
        best_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for step, batch in enumerate(loop):
                inputs = prepare_msn_inputs(batch)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                loss_value = handle_msn_training_step(
                    model=self.model,
                    inputs=inputs,
                    loss_fn=self.loss_fn,
                    optimizer=self.optimizer,
                    global_step=self.global_step,
                    artifact_dumper=self.artifact_dumper,
                    log_every=self.log_every,
                    epoch=epoch,
                    step=step,
                )

                loop.set_postfix({"loss": loss_value})
                self.global_step += 1

            if self.scheduler:
                if isinstance(self.scheduler, dict):
                    if "generator" in self.scheduler:
                        self.scheduler["generator"].step()
                    if "discriminator" in self.scheduler:
                        self.scheduler["discriminator"].step()
                else:
                    self.scheduler.step()

            val_loss = self.evaluate()
            if val_loss < best_loss:
                best_loss = val_loss
                self.save(suffix="best_model")
                print(f"New best model saved with loss: {best_loss:.4f}")

            self.save(suffix="latest")

        if self.artifact_dumper:
            self.artifact_dumper.save(filename=f"msn_final_epoch{num_epochs}.pt")

        return {"final_loss": val_loss, "best_loss": best_loss}

    def evaluate(self) -> float:
        """
        Evaluate the MSN model on the validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs = self._prepare_msn_inputs(batch)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                output = self.model(inputs)
                loss_output: LossOutput = self.loss_fn(output)

                total_loss += loss_output.total.item()
                count += 1

                if (
                    self.artifact_dumper
                    and self.log_every
                    and count % self.log_every == 0
                ):
                    self.artifact_dumper.log_full_output(
                        output,
                        loss=loss_output,
                        step=self.global_step,
                        batch_id=f"val_step{count}",
                        prefix="val",
                    )

        avg_loss = total_loss / count if count > 0 else float("inf")
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def _prepare_msn_inputs(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Convert batch to dictionary format expected by MSNWrapper.

        Args:
            batch (Any): Batch from DataLoader.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'anchor' and 'target' keys.

        Raises:
            TypeError: If the batch format is unsupported.
        """
        if isinstance(batch, dict) and "anchor" in batch and "target" in batch:
            return batch
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            return {"anchor": batch[0], "target": batch[1]}
        elif isinstance(batch, torch.Tensor):
            return {"anchor": batch, "target": batch}
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
