"""
AETrainer implementation for autoencoder-based unsupervised learning tasks.

This module defines the AETrainer class, which handles training and evaluation
of autoencoder models. It supports logging, artifact dumping, and checkpointing.
"""

from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.trainer.utils.autoencoder_utils import (
    extract_autoencoder_inputs,
    handle_autoencoder_evaluation_step,
    handle_autoencoder_training_step,
)

T = TypeVar("T", bound=torch.Tensor)


@register_trainer("autoencoder")
class AETrainer(BaseTrainer):
    """
    Trainer for autoencoder-based unsupervised learning tasks.

    Handles training, evaluation, logging, and artifact dumping for autoencoder models.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        loss_fn: Callable[[ModelOutput, torch.Tensor], LossOutput],
        optimizer_cls: Callable[..., Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        artifact_dumper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the AETrainer.

        Args:
            model (Module): The autoencoder model to be trained.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            loss_fn (Callable): Loss function for autoencoder training.
            optimizer_cls (Callable[..., Optimizer]): Optimizer class.
            optimizer_args (Optional[Dict[str, Any]]): Arguments for the optimizer.
            device (str, optional): Device to use (default: "cuda").
            scheduler (Optional[Any], optional): Learning rate scheduler.
            artifact_dumper (Optional[Any], optional): Artifact logger/dumper.
            **kwargs: Additional keyword arguments.
        """
        # Ensure model_name is properly set from kwargs
        if "model_name" not in kwargs:
            # Fallback to the old behavior if model_name is not provided
            variant = kwargs.pop("model_variant", "simple")
            kwargs["model_name"] = f"autoencoder_{variant}"
        
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
        self.log_every = (
            getattr(artifact_dumper, "log_every", 1) if artifact_dumper else None
        )  # Changed to 1 for every step
        self.global_step = 0

        if optimizer_args is None:
            optimizer_args = {"lr": 1e-3}

        self.logger = self._get_logger()
        self.optimizer: Optional[Union[Optimizer, Dict[str, Optimizer]]] = (
            optimizer_cls(self.model.parameters(), **optimizer_args)
        )

    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        Train the autoencoder model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
        """
        best_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for _step, batch in enumerate(loop):
                inputs = extract_autoencoder_inputs(batch)
                inputs = inputs.to(self.device)

                loss_value, output = handle_autoencoder_training_step(
                    model=self.model,
                    inputs=inputs,
                    loss_fn=self.loss_fn,
                    optimizer=self.optimizer,
                    global_step=self.global_step,
                    artifact_dumper=self.artifact_dumper,
                )

                self.global_step += 1
                loop.set_postfix({"loss": loss_value})

            # Validation
            val_loss = self.evaluate()

            if val_loss < best_loss:
                best_loss = val_loss
                self.save(suffix="best_model")
                print(f"New best model saved with loss: {val_loss:.4f}")

            self.save(suffix="latest")

        return {"final_loss": best_loss, "best_loss": best_loss}

    def evaluate(self) -> float:
        """
        Evaluate the autoencoder model on the validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)

            for val_step, batch in enumerate(loop):
                # Use a separate step counter for validation to avoid conflicts
                val_global_step = (
                    self.global_step + val_step + 1000000
                )  # Large offset to avoid conflicts

                inputs = extract_autoencoder_inputs(batch)
                inputs = inputs.to(self.device)

                loss_value = handle_autoencoder_evaluation_step(
                    model=self.model,
                    inputs=inputs,
                    loss_fn=self.loss_fn,
                    global_step=val_global_step,
                    artifact_dumper=self.artifact_dumper,
                )

                total_loss += loss_value

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def _unwrap_output(
        self, output: Union[ModelOutput, Dict[str, Any], torch.Tensor]
    ) -> ModelOutput:
        """
        Convert output to ModelOutput if not already.

        Args:
            output (Union[ModelOutput, Dict[str, Any], torch.Tensor]): Model output.

        Returns:
            ModelOutput: Wrapped model output.

        Raises:
            ValueError: If output is None.
        """
        if output is None:
            raise ValueError("[_unwrap_output] Received None as output!")

        if isinstance(output, ModelOutput):
            return output
        elif isinstance(output, dict):
            return ModelOutput(**output)
        else:
            return ModelOutput(reconstruction=output)

    def _extract_inputs(
        self, batch: Union[torch.Tensor, Dict[str, Any], list[Any], tuple[Any, ...]]
    ) -> torch.Tensor:
        """
        Extract input tensor from a batch.

        Args:
            batch (Union[torch.Tensor, Dict[str, Any], list, tuple]): Batch from DataLoader.

        Returns:
            torch.Tensor: Input tensor.

        Raises:
            TypeError: If input tensor cannot be extracted.
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) == 0 or not isinstance(batch[0], torch.Tensor):
                raise TypeError(
                    "Batch is empty or does not contain a tensor as the first element."
                )
            return batch[0]
        if isinstance(batch, dict):
            image = batch.get("image")
            if image is not None and isinstance(image, torch.Tensor):
                return cast(torch.Tensor, image)
            input_tensor = batch.get("input")
            if input_tensor is not None and isinstance(input_tensor, torch.Tensor):
                return cast(torch.Tensor, input_tensor)
            raise TypeError(
                "Batch dict does not contain a valid 'image' or 'input' tensor."
            )
        if isinstance(batch, torch.Tensor):
            return batch
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _get_logger(self) -> Optional[Any]:
        """
        Retrieve the logger from the artifact dumper if available.

        Returns:
            Optional[Any]: Logger object if available, else None.
        """
        if self.artifact_dumper and hasattr(self.artifact_dumper, "logger"):
            return self.artifact_dumper.logger
        return None
