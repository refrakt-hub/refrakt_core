from typing import Any, Callable, Dict, Optional, Union

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
    from refrakt_xai.utils import generate_explainability
except ImportError:
    generate_explainability = None

try:
    from refrakt_viz.utils import visualize_embeddings, visualize_attention
except ImportError:
    visualize_embeddings = visualize_attention = None


@register_trainer("supervised")
class SupervisedTrainer(BaseTrainer):
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        optimizer_cls: Callable[..., Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        artifact_dumper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
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
        if isinstance(optimizer_args, DictConfig):
            optimizer_args = OmegaConf.to_container(optimizer_args, resolve=True)
        self.optimizer = optimizer_cls(self.model.parameters(), **(optimizer_args or {"lr": 1e-4}))

        self.grad_log_interval = kwargs.get("grad_log_interval", 100)
        self.param_log_interval = kwargs.get("param_log_interval", 500)
        self.log_every = getattr(self.artifact_dumper, "log_every", 10) if self.artifact_dumper else None
        self.global_step = 0

    def train(self, num_epochs: int) -> Dict[str, float]:
        best_accuracy = 0.0
        logger = self._get_logger()
        
        # Log initial parameters
        if logger and self.global_step == 0:
            logger.log_parameters(self.model, step=self.global_step, prefix="init_")

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(loop):
                inputs, targets = self._unpack_batch(batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss_output: LossOutput = self.loss_fn(output, targets)
                loss_output.total.backward()
                
                # Log gradients and parameters periodically
                if logger and self.global_step % self.grad_log_interval == 0:
                    logger.log_gradients(self.model, step=self.global_step, prefix="grads/")
                    
                if logger and self.global_step % self.param_log_interval == 0:
                    logger.log_parameters(self.model, step=self.global_step, prefix="params/")
                    
                    # Log learning rate
                    lr = self.optimizer.param_groups[0]['lr']
                    logger.log_metrics({"lr": lr}, step=self.global_step)
                
                self.optimizer.step()
                
                # Log loss components
                loss_summary = loss_output.summary()
                if self.artifact_dumper:
                    self.artifact_dumper.log_scalar_dict(loss_summary, step=self.global_step, prefix="train/loss")
                
                # Log additional metrics
                if isinstance(output, ModelOutput) and hasattr(output, "summary"):
                    output_summary = output.summary()
                    if self.artifact_dumper:
                        self.artifact_dumper.log_scalar_dict(output_summary, step=self.global_step, prefix="train/output")
                
                loop.set_postfix({"loss": loss_output.total.item()})

                # Artifact logging
                if self.artifact_dumper and self.artifact_dumper.should_log_step(self.global_step):
                    self.artifact_dumper.log_output(output, batch_id=self.global_step, targets=targets)
                    
                    # Log input samples (first batch of each epoch)
                    if step == 0:
                        self.artifact_dumper.log_output(
                            ModelOutput(image=inputs), 
                            batch_id=f"epoch{epoch}_inputs"
                        )

                self.global_step += 1

            if self.scheduler:
                self.scheduler.step()
                print(f"Epoch {epoch + 1} complete. LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            acc = self.evaluate()
            if acc > best_accuracy:
                best_accuracy = acc
                self.save(suffix="best_model")
                print(f"New best model saved with accuracy: {acc * 100:.2f}%")

            self.save(suffix="latest")

        # Log final parameters
        if logger:
            logger.log_parameters(self.model, step=self.global_step, prefix="final_")
            
        # Return final metrics
        return {
            "best_accuracy": best_accuracy,
            "final_loss": loss_output.total.item() if loss_output else 0.0,
            "total_steps": self.global_step
        }

    def evaluate(self) -> float:
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)

            for batch in loop:
                inputs, targets = self._unpack_batch(batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                output = self.model(inputs)
                logits = output.logits if isinstance(output, ModelOutput) else output
                preds = torch.argmax(logits, dim=1)

                correct += (preds == targets).sum().item()
                total += targets.size(0)
                loop.set_postfix({"acc": f"{(correct / total * 100):.2f}%"})

        acc = correct / total if total > 0 else 0.0
        print(f"\nValidation Accuracy: {acc * 100:.2f}%")

        # Log validation accuracy
        if self.artifact_dumper:
            self.artifact_dumper.log_scalar_dict(
                {"accuracy": acc}, 
                step=self.global_step, 
                prefix="val"
            )

        return acc

    def _unpack_batch(self, batch: Union[tuple, list, Dict[str, torch.Tensor]]) -> tuple:
        if isinstance(batch, (tuple, list)):
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch["input"], batch["target"]
        raise TypeError("Unsupported batch format")
    
    def _get_logger(self):
        """Helper to get logger from artifact dumper or extra params"""
        if self.artifact_dumper and hasattr(self.artifact_dumper, 'logger'):
            return self.artifact_dumper.logger
        return self.extra_params.get('logger')