"""
SupervisedTrainer implementation for standard supervised learning tasks.

This module defines the SupervisedTrainer class, which handles training and evaluation
of models using supervised objectives (e.g., classification, regression).
It supports logging, artifact dumping, and integration with explainability/visualization tools.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.trainer.utils.supervised_utils import (
    handle_epoch_end,
    handle_training_step,
    log_artifacts,
    log_training_metrics,
)

try:
    from refrakt_xai.utils import generate_explainability  # type: ignore
except ImportError:
    generate_explainability = None

try:
    from refrakt_viz.utils import (  # type: ignore[import-not-found]
        visualize_attention,
        visualize_embeddings,
    )
except ImportError:
    visualize_embeddings = visualize_attention = None  # type: ignore[unused-variable]


@register_trainer("supervised")
class SupervisedTrainer(BaseTrainer):
    """
    Supervised trainer for classification and regression tasks.
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
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            device=device,
            scheduler=scheduler,
            artifact_dumper=artifact_dumper,
            **kwargs,
        )

        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.extra_params = kwargs
        self.grad_log_interval = kwargs.get("grad_log_interval", 100)
        self.param_log_interval = kwargs.get("param_log_interval", 500)
        self.log_every = (
            getattr(self.artifact_dumper, "log_every", 10)
            if self.artifact_dumper
            else None
        )
        self.global_step = 0
        self._current_batch = None
        self._current_loss_output = None

        if self.optimizer is None:
            from omegaconf import DictConfig

            args = optimizer_args
            if isinstance(args, DictConfig):
                pass
            final_args = args or {"lr": 1e-3}
            self.optimizer = optimizer_cls(self.model.parameters(), **final_args)

    def _handle_training_step(self, batch: Any, step: int, epoch: int) -> None:
        """Handle a single training step."""
        return handle_training_step(self, batch, step, epoch)

    def _log_training_metrics(self, loss_output: Any, output: Any, step: int) -> None:
        """Log training metrics."""
        return log_training_metrics(self, loss_output, output, step)

    def _log_artifacts(
        self, output: Any, loss_output: Any, step: int, epoch: int
    ) -> None:
        """Log artifacts for the current step."""
        return log_artifacts(self, output, loss_output, step, epoch)

    def _handle_epoch_end(self, epoch: int, best_accuracy: float) -> float:
        """Handle end of epoch operations."""
        return handle_epoch_end(self, epoch, best_accuracy)

    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
        """
        best_accuracy = 0.0
        logger = self._get_logger()

        if logger and self.global_step == 0:
            logger.log_parameters(self.model, step=self.global_step, prefix="init_")

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(loop):
                self._current_batch = batch  # Store for artifact logging
                self._handle_training_step(batch, step, epoch)
                if self._current_loss_output is not None:
                    pass

            best_accuracy = self._handle_epoch_end(epoch, best_accuracy)

        if logger:
            logger.log_parameters(self.model, step=self.global_step, prefix="final_")

        return {"best_accuracy": best_accuracy}

    def evaluate(self) -> float:
        """
        Evaluate the model on the validation set.

        Returns:
            float: Validation accuracy (0.0 if no samples).
        """
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)

            for batch in loop:
                inputs, targets = self._unpack_batch(batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                output = self.model(inputs)
                # Extract logits for predictions (don't call loss function!)
                if isinstance(output, ModelOutput) and output.logits is not None:
                    logits = output.logits
                elif isinstance(output, torch.Tensor):
                    logits = output
                else:
                    raise ValueError(
                        "Output does not have logits for argmax in evaluate()."
                    )

                if logits is not None:
                    preds = torch.argmax(logits, dim=1)
                else:
                    raise ValueError("Logits are None in evaluate().")

                correct += (preds == targets).sum().item()
                total += targets.size(0)
                loop.set_postfix({"acc": f"{(correct / total * 100):.2f}%"})

        acc = correct / total if total > 0 else 0.0
        print(f"\nValidation Accuracy: {acc * 100:.2f}%")

        if self.artifact_dumper:
            self.artifact_dumper.log_scalar_dict(
                {"accuracy": acc}, step=self.global_step, prefix="val"
            )

        return acc

    def _unpack_batch(
        self, batch: Union[Tuple[Any, Any], List[Any], Dict[str, torch.Tensor]]
    ) -> Tuple[Any, Any]:
        """
        Unpack a batch into input and target tensors.

        Args:
            batch (Union[tuple, list, Dict[str, torch.Tensor]]): Batch from DataLoader.

        Returns:
            tuple: (inputs, targets)

        Raises:
            TypeError: If the batch format is unsupported.
        """
        if isinstance(batch, (tuple, list)):
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch["input"], batch["target"]
        raise TypeError("Unsupported batch format")

    def _get_logger(self) -> Optional[Any]:
        """
        Retrieve the logger from the artifact dumper or extra parameters.

        Returns:
            Any: Logger object if available, else None.
        """
        if self.artifact_dumper and hasattr(self.artifact_dumper, "logger"):
            return self.artifact_dumper.logger
        return self.extra_params.get("logger")
