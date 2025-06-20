from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.registry.trainer_registry import register_trainer

@register_trainer("msn")
class MSNTrainer(BaseTrainer):
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
        # Initialize with artifact_dumper in base class
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
        self.artifact_dumper = artifact_dumper
        self.extra_params = kwargs

        # Optimizer is built using passed class and arguments
        self.optimizer = optimizer_cls(
            self.model.parameters(), 
            **(optimizer_args or {"lr": 1e-3})
        )
        
        # Logging setup
        self.log_every = getattr(artifact_dumper, "log_every", 10) if artifact_dumper else None
        self.global_step = 0

    def train(self, num_epochs: int) -> Dict[str, float]:
        best_loss = float('inf')
        metrics = {}

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for step, batch in enumerate(loop):
                # Prepare inputs (handles both dict and tuple formats)
                inputs = self._prepare_msn_inputs(batch)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(inputs)
                
                # Compute loss
                loss_output: LossOutput = self.loss_fn(output)
                loss_output.total.backward()
                self.optimizer.step()
                
                # Update progress
                loop.set_postfix({"loss": loss_output.total.item()})
                
                # Log artifacts
                if self.artifact_dumper and self.log_every and step % self.log_every == 0:
                    self.artifact_dumper.log_full_output(
                        output,
                        loss=loss_output,
                        step=self.global_step,
                        batch_id=f"train_ep{epoch}_step{step}"
                    )
                
                self.global_step += 1

            if self.scheduler:
                self.scheduler.step()

            # Validation and checkpointing
            val_loss = self.evaluate()
            if val_loss < best_loss:
                best_loss = val_loss
                self.save(suffix="best_model")
                print(f"New best model saved with loss: {best_loss:.4f}")

            self.save(suffix="latest")

        metrics = {
            "best_val_loss": best_loss,
            "global_step": self.global_step
        }
        
        if self.artifact_dumper:
            self.artifact_dumper.save(filename=f"msn_final_epoch{num_epochs}.pt")
        
        return metrics

    def evaluate(self) -> float:
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
                
                # Log validation artifacts
                if self.artifact_dumper and self.log_every and count % self.log_every == 0:
                    self.artifact_dumper.log_full_output(
                        output,
                        loss=loss_output,
                        step=self.global_step,
                        batch_id=f"val_step{count}",
                        prefix="val"
                    )

        avg_loss = total_loss / count if count > 0 else float('inf')
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def _prepare_msn_inputs(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Convert batch to dictionary format expected by MSNWrapper"""
        if isinstance(batch, dict) and 'anchor' in batch and 'target' in batch:
            return batch
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            return {'anchor': batch[0], 'target': batch[1]}
        elif isinstance(batch, torch.Tensor):
            # Use same tensor for both anchor and target
            return {'anchor': batch, 'target': batch}
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")