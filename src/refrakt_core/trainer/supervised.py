from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer


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
        super().__init__(model, train_loader, val_loader, device, **kwargs)
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.extra_params = kwargs

        from omegaconf import DictConfig, OmegaConf
        if isinstance(optimizer_args, DictConfig):
            optimizer_args = OmegaConf.to_container(optimizer_args, resolve=True)
        self.optimizer = optimizer_cls(self.model.parameters(), **(optimizer_args or {"lr": 1e-4}))

        self.artifact_dumper = artifact_dumper
        self.log_every = getattr(self.artifact_dumper, "log_every", 10) if self.artifact_dumper else None

    def train(self, num_epochs: int) -> None:
        best_accuracy = 0.0

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
                self.optimizer.step()
                loop.set_postfix({"loss": loss_output.total.item()})

                # === Artifact logging every N steps
                if self.artifact_dumper and self.log_every and step % self.log_every == 0:
                    if isinstance(output, ModelOutput):
                        self.artifact_dumper.log_output(output, batch_id=f"epoch{epoch}_step{step}")
                    self.artifact_dumper.log_loss(loss_output, batch_id=f"epoch{epoch}_step{step}")

            if self.scheduler:
                self.scheduler.step()
                print(f"Epoch {epoch + 1} complete. LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            acc = self.evaluate()
            if acc > best_accuracy:
                best_accuracy = acc
                self.save(suffix="best_model")
                print(f"New best model saved with accuracy: {acc * 100:.2f}%")

            self.save(suffix="latest")

        # Final artifact save
        if self.artifact_dumper:
            self.artifact_dumper.save(filename=f"train_epoch{num_epochs}_final.pt")

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
        return acc

    def _unpack_batch(self, batch: Union[tuple, list, Dict[str, torch.Tensor]]) -> tuple:
        if isinstance(batch, (tuple, list)):
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch["input"], batch["target"]
        raise TypeError("Unsupported batch format")
