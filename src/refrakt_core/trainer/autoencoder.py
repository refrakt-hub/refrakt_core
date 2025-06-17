from typing import Any, Callable, Dict, Optional, Union
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.wrappers.losses.mae import MAELossWrapper


@register_trainer("autoencoder")
class AETrainer(BaseTrainer):
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable[[ModelOutput, torch.Tensor], LossOutput],
        optimizer_cls: Callable[..., Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        artifact_dumper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        variant = kwargs.pop("model_variant", "simple")
        kwargs["model_name"] = f"autoencoder_{variant}"
        super().__init__(model, train_loader, val_loader, device, artifact_dumper=artifact_dumper, **kwargs)

        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.log_every = getattr(artifact_dumper, "log_every", 1) if artifact_dumper else None  # Changed to 1 for every step
        self.global_step = 0

        if optimizer_args is None:
            optimizer_args = {"lr": 1e-3}

        self.logger = self._get_logger()
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)

    def train(self, num_epochs: int) -> Dict[str, float]:
        best_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(loop):
                inputs = self._extract_inputs(batch).to(self.device)
                
                # Reshape inputs if they're flattened
                if inputs.dim() == 2 and hasattr(self.model, 'expected_input_dim'):
                    # Only reshape if model specifies expected dimensions
                    inputs = inputs.view(-1, *self.model.expected_input_dim)
                    
                self.optimizer.zero_grad()
                output = self.model(inputs)
                
                # In train() method, after getting output:
                if not isinstance(output, ModelOutput):
                    output = self._unwrap_output(output)
                    
                # In the train() method, replace the loss computation with:
                loss_output = self.loss_fn(output, inputs if not isinstance(self.loss_fn, MAELossWrapper) else None)
                    
                loss_output.total.backward()
                self.optimizer.step()

                # Log training metrics at every step
                if self.artifact_dumper:
                    self.artifact_dumper.log_scalar_dict(
                        loss_output.summary(), 
                        step=self.global_step, 
                        prefix="train"
                    )

                    if isinstance(output, ModelOutput):
                        self.artifact_dumper.log_scalar_dict(
                            output.summary(), 
                            step=self.global_step, 
                            prefix="train"
                        )

                self.global_step += 1
                loop.set_postfix({"loss": loss_output.total.item()})

            # Validation - no visualization, just metrics
            val_loss = self.evaluate()
            
            if val_loss < best_loss:
                best_loss = val_loss
                self.save(suffix="best_model")
                print(f"New best model saved with loss: {val_loss:.4f}")

            self.save(suffix="latest")

        return {
            "best_val_loss": best_loss,
            "total_steps": self.global_step
        }

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)

            for val_step, batch in enumerate(loop):
                # Use a separate step counter for validation to avoid conflicts
                val_global_step = self.global_step + val_step + 1000000  # Large offset to avoid conflicts

                inputs = self._extract_inputs(batch).to(self.device)

                if inputs.dim() == 2 and hasattr(self.model, 'expected_input_dim'):
                    # Only reshape if model specifies expected dimensions
                    inputs = inputs.view(-1, *self.model.expected_input_dim)
                    
                output = self.model(inputs)
                loss_output: LossOutput = self.loss_fn(output, inputs)
                total_loss += loss_output.total.item()

                # Log validation metrics at every step with unique step counter
                if self.artifact_dumper:
                    self.artifact_dumper.log_scalar_dict(
                        loss_output.summary(),
                        step=val_global_step,
                        prefix="val"
                    )

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def _unwrap_output(self, output: Union[ModelOutput, Dict, torch.Tensor]) -> ModelOutput:
        if output is None:
            raise ValueError("[_unwrap_output] Received None as output!")

        if isinstance(output, ModelOutput):
            return output
        elif isinstance(output, dict):
            return ModelOutput(**output)
        else:
            return ModelOutput(reconstruction=output)

    def _extract_inputs(self, batch: Union[torch.Tensor, Dict, list, tuple]) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            return batch[0]
        if isinstance(batch, dict):
            return batch.get("image") or batch.get("input")
        return batch

    def _get_logger(self):
        if self.artifact_dumper and hasattr(self.artifact_dumper, "logger"):
            return self.artifact_dumper.logger
        return None