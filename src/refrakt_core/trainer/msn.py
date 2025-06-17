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
    """
    Trainer for Masked Siamese Networks (MSN) models.
    
    Handles the unique training requirements of MSN models, including:
    - Processing of masked and unmasked patches
    - Custom loss functions for siamese networks
    - Specialized artifact logging for visualizations
    """
    
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
        variant = kwargs.pop("model_variant", "msn")
        kwargs["model_name"] = f"msn_{variant}"
        super().__init__(model, train_loader, val_loader, device, **kwargs)

        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.artifact_dumper = artifact_dumper
        self.extra_params = kwargs

        from omegaconf import DictConfig, OmegaConf
        if isinstance(optimizer_args, DictConfig):
            optimizer_args = OmegaConf.to_container(optimizer_args, resolve=True)

        self.optimizer = optimizer_cls(self.model.parameters(), **(optimizer_args or {"lr": 1e-3}))
        self.log_every = getattr(self.artifact_dumper, "log_every", 10) if self.artifact_dumper else None

    def train(self, num_epochs: int) -> None:
        best_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(loop):
                # MSN models typically require both masked and unmasked inputs
                inputs = self._prepare_msn_inputs(batch).to(self.device)

                self.optimizer.zero_grad()
                raw_outputs = self.model(inputs)
                model_output = self._build_model_output(raw_outputs, inputs)

                # MSN loss typically uses predictions and targets from siamese branches
                loss_output: LossOutput = self.loss_fn(model_output)
                loss_output.total.backward()
                self.optimizer.step()

                loop.set_postfix({"loss": loss_output.total.item()})

                # Artifact logging for visualization
                if self.artifact_dumper and self.log_every and step % self.log_every == 0:
                    self.artifact_dumper.log_output(model_output, batch_id=f"train_ep{epoch}_step{step}")
                    self.artifact_dumper.log_loss(loss_output, batch_id=f"train_ep{epoch}_step{step}")

            if self.scheduler:
                self.scheduler.step()

            val_loss = self.evaluate()
            if val_loss < best_loss:
                best_loss = val_loss
                self.save(suffix="best_model")
                print(f"New best model saved with loss: {best_loss:.4f}")

            self.save(suffix="latest")

        if self.artifact_dumper:
            self.artifact_dumper.save(filename=f"msn_final_epoch{num_epochs}.pt")

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
                inputs = self._prepare_msn_inputs(batch).to(self.device)

                raw_outputs = self.model(inputs)
                model_output = self._build_model_output(raw_outputs, inputs)
                
                loss_output: LossOutput = self.loss_fn(model_output)
                total_loss += loss_output.total.item()

                if self.artifact_dumper and self.log_every and step % self.log_every == 0:
                    self.artifact_dumper.log_output(model_output, batch_id=f"val_step{step}")
                    self.artifact_dumper.log_loss(loss_output, batch_id=f"val_step{step}")

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def _prepare_msn_inputs(self, batch: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """Prepare inputs for MSN training, handling both masked and unmasked patches"""
        if isinstance(batch, dict):
            # MSN batches typically contain both masked and unmasked patches
            return {k: v.to(self.device) for k, v in batch.items()}
        return batch.to(self.device)

    def _build_model_output(self, output: Union[Dict, torch.Tensor], inputs: Any) -> ModelOutput:
        """Construct a standardized ModelOutput object from MSN model results"""
        if isinstance(output, dict):
            return ModelOutput(
                embeddings=output.get("embeddings"),
                targets=inputs.get("unmasked") if isinstance(inputs, dict) else inputs,
                attention_maps=output.get("attention_maps"),
                extra={
                    "masked_embeddings": output.get("masked_embeddings"),
                    "unmasked_embeddings": output.get("unmasked_embeddings"),
                    "mask": output.get("mask"),
                    "predicted_mask": output.get("predicted_mask"),
                }
            )
        else:
            return ModelOutput(
                embeddings=output,
                targets=inputs,
                extra={"raw": output},
            )