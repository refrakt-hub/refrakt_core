"""
SupervisedTrainer implementation for standard supervised learning tasks.

This module defines the SupervisedTrainer class, which handles training and evaluation
of models using supervised objectives (e.g., classification, regression).
It supports logging, artifact dumping, and integration with explainability/visualization tools.
"""
from typing import Any, Callable, Dict, Optional, Union, Tuple, List, cast
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer

try:
    from refrakt_xai.utils import generate_explainability  # type: ignore[import-not-found]
except ImportError:
    generate_explainability = None

try:
    from refrakt_viz.utils import visualize_embeddings, visualize_attention  # type: ignore[import-not-found]
except ImportError:
    visualize_embeddings = visualize_attention = None


@register_trainer("supervised")
class SupervisedTrainer(BaseTrainer):
    """
    Trainer for supervised learning tasks (classification, regression, etc.).

    Handles training, evaluation, logging, and artifact dumping for supervised models.
    Optionally integrates with explainability and visualization tools if available.
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
        Initialize the SupervisedTrainer.

        Args:
            model (Module): The model to be trained.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            loss_fn (Callable): Loss function for supervised learning.
            optimizer_cls (Callable[..., Optimizer]): Optimizer class.
            optimizer_args (Optional[Dict[str, Any]]): Arguments for the optimizer.
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
            **kwargs
        )
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.extra_params = kwargs

        from omegaconf import DictConfig, OmegaConf
        if isinstance(optimizer_args, DictConfig): # type: ignore[unreachable]
            _tmp_args = OmegaConf.to_container(optimizer_args, resolve=True)
            if isinstance(_tmp_args, dict):
                optimizer_args = cast(Dict[str, Any], _tmp_args)
            else:
                optimizer_args = None
        self.optimizer: Optional[Union[Optimizer, Dict[str, Optimizer]]] = optimizer_cls(self.model.parameters(), **(optimizer_args or {"lr": 1e-4}))

        self.grad_log_interval = kwargs.get("grad_log_interval", 100)
        self.param_log_interval = kwargs.get("param_log_interval", 500)
        self.log_every = getattr(self.artifact_dumper, "log_every", 10) if self.artifact_dumper else None
        self.global_step = 0

    def train(self, num_epochs: int) -> None:
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
                inputs, targets = self._unpack_batch(batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if self.optimizer is not None:
                    if isinstance(self.optimizer, Optimizer):
                        self.optimizer.zero_grad()
                output = self.model(inputs)
                logits = output.logits if isinstance(output, ModelOutput) else output

                loss = self.loss_fn(logits, targets)
                loss_output = LossOutput(total=loss, components={"main": loss})

                assert isinstance(loss_output.total, torch.Tensor)
                loss_output.total.backward()  # type: ignore[no-untyped-call]
                
                if logger and self.global_step % self.grad_log_interval == 0:
                    logger.log_gradients(self.model, step=self.global_step, prefix="")
                if logger and self.global_step % self.param_log_interval == 0:
                    logger.log_parameters(self.model, step=self.global_step, prefix="")
                    if self.optimizer is not None and isinstance(self.optimizer, Optimizer):
                        lr = self.optimizer.param_groups[0]['lr']
                        logger.log_metrics({"lr": lr}, step=self.global_step)
                
                if self.optimizer is not None:
                    if isinstance(self.optimizer, Optimizer):
                        self.optimizer.step()
                
                loss_summary = loss_output.summary()
                if self.artifact_dumper:
                    self.artifact_dumper.log_scalar_dict(loss_summary, step=self.global_step, prefix="train")
                
                if isinstance(output, ModelOutput) and hasattr(output, "summary"):
                    output_summary = output.summary()
                    if self.artifact_dumper:
                        self.artifact_dumper.log_scalar_dict(output_summary, step=self.global_step, prefix="train/output")
                
                loop.set_postfix({"loss": loss_output.total.item()})

                # === Artifact logging ===
                if self.artifact_dumper and self.artifact_dumper.should_log_step(self.global_step):
                    # wrap everything inside a ModelOutput
                    full_output = output
                    if not isinstance(output, ModelOutput):
                        full_output = ModelOutput(logits=output)
                    full_output.targets = targets
                    full_output.image = inputs

                    self.artifact_dumper.log_full_output(
                        full_output,
                        loss=loss_output,
                        step=self.global_step,
                        batch_id=f"step{self.global_step}"
                    )

                    if step == 0:
                        self.artifact_dumper.log_full_output(
                            full_output,
                            loss=loss_output,
                            step=self.global_step,
                            batch_id=f"epoch{epoch}_step{step}"
                        )

                self.global_step += 1

            if self.scheduler and not isinstance(self.scheduler, dict) and hasattr(self.scheduler, 'step'):
                self.scheduler.step()
                if self.optimizer is not None and isinstance(self.optimizer, Optimizer):
                    print(f"Epoch {epoch + 1} complete. LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            acc = self.evaluate()
            if acc > best_accuracy:
                best_accuracy = acc
                self.save(suffix="best_model")
                print(f"New best model saved with accuracy: {acc * 100:.2f}%")

            self.save(suffix="latest")

        if logger:
            logger.log_parameters(self.model, step=self.global_step, prefix="final_")

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
                logits = output.logits if isinstance(output, ModelOutput) else output
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
                {"accuracy": acc}, 
                step=self.global_step, 
                prefix="val"
            )

        return acc

    def _unpack_batch(self, batch: Union[Tuple[Any, Any], List[Any], Dict[str, torch.Tensor]]) -> Tuple[Any, Any]:
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
        if self.artifact_dumper and hasattr(self.artifact_dumper, 'logger'):
            return self.artifact_dumper.logger
        return self.extra_params.get('logger')
