from typing import Any, Dict, Optional, Union, Callable

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer


@register_trainer("gan")
class GANTrainer(BaseTrainer):
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Dict[str, Callable],
        optimizer_cls: Dict[str, Callable[..., Optimizer]],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        artifact_dumper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, train_loader, val_loader, device, **kwargs)

        self.loss_fns = loss_fn
        self.scheduler = scheduler
        self.artifact_dumper = artifact_dumper
        self.log_every = getattr(self.artifact_dumper, "log_every", 10) if self.artifact_dumper else None

        self.optimizer = {
            key: optimizer_cls[key](self.model.get_submodule(key).parameters(), **(optimizer_args or {"lr": 1e-4}))
            for key in ["generator", "discriminator"]
        }

    def train(self, num_epochs: int) -> None:
        best_metric = float("-inf")

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(loop):
                device_batch = self._move_batch_to_device(batch)

                losses = self.model.training_step(
                    device_batch,
                    optimizer=self.optimizer,
                    loss_fn=self.loss_fns,
                    device=self.device,
                )

                loop.set_postfix({
                    "gen_loss": losses.get("g_loss", 0),
                    "disc_loss": losses.get("d_loss", 0),
                })

                if self.artifact_dumper and self.log_every and step % self.log_every == 0:
                    # Create comprehensive loss output
                    loss_output = LossOutput(
                        total=losses.get("g_loss", 0) + losses.get("d_loss", 0),
                        components={k: v for k, v in losses.items()}
                    )
                    self.artifact_dumper.log_loss(
                        loss_output,
                        batch_id=f"epoch{epoch}_step{step}",
                    )

            if self.scheduler:
                self.scheduler["generator"].step()
                self.scheduler["discriminator"].step()

            metric = self.evaluate()
            if metric > best_metric:
                best_metric = metric
                self.save(suffix="best_model")
                print(f"New best model saved with PSNR: {best_metric:.2f} dB")

            self.save(suffix="latest")

        if self.artifact_dumper:
            self.artifact_dumper.save(filename=f"gan_final_epoch{num_epochs}.pt")

    def evaluate(self) -> float:
        self.model.eval()
        total_psnr = 0.0

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.val_loader, desc="Evaluating", leave=False)):
                if isinstance(batch, dict):
                    lr = batch.get("lr", batch.get("input"))
                    hr = batch.get("hr", batch.get("target"))
                else:
                    lr, hr = batch[0], batch[1]

                lr = lr.to(self.device)
                hr = hr.to(self.device)
                sr = self.model.generate(lr)

                # Calculate PSNR
                mse = torch.mean((sr - hr) ** 2)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                total_psnr += psnr.item()

                if self.artifact_dumper and self.log_every and step % self.log_every == 0:
                    model_output = ModelOutput(
                        image=sr,
                        targets=hr,
                        extra={"low_res": lr},
                    )
                    self.artifact_dumper.log_output(model_output, batch_id=f"val_step{step}")

        avg_psnr = total_psnr / len(self.val_loader)
        print(f"\nValidation PSNR: {avg_psnr:.2f} dB")
        return avg_psnr

    def _move_batch_to_device(
        self, batch: Union[Dict[str, torch.Tensor], list, tuple]
    ) -> Union[Dict[str, torch.Tensor], list[torch.Tensor]]:
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        return [x.to(self.device) for x in batch]