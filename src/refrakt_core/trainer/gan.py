"""
Trainer module for Generative Adversarial Networks (GANs).

Handles training and evaluation logic for GAN models using a structured interface
for generator/discriminator, loss functions, and optimizers.
"""

from typing import Any, Dict, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer


@register_trainer("gan")
class GANTrainer(BaseTrainer):
    """
    GAN Trainer for adversarial training.

    Args:
        model (Module): The GAN model (must implement `training_step` and `generate`).
        train_loader (DataLoader): DataLoader for training.
        val_loader (DataLoader): DataLoader for validation.
        loss_fn (Dict[str, Callable]): Dict containing "generator" and "discriminator" loss functions.
        optimizer (Dict[str, Optimizer]): Dict containing "generator" and "discriminator" optimizers.
        device (str): Device for training ("cuda" or "cpu").
        scheduler (Optional[Any]): Optional learning rate scheduler.
        **kwargs: Extra parameters passed to BaseTrainer.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Dict[str, Any],
        optimizer: Dict[str, Optimizer],
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, train_loader, val_loader, device, **kwargs)

        if not {"generator", "discriminator"}.issubset(loss_fn):
            raise ValueError("loss_fn must contain 'generator' and 'discriminator' keys")

        if not {"generator", "discriminator"}.issubset(optimizer):
            raise ValueError("optimizer must contain 'generator' and 'discriminator' keys")

        self.loss_fns = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, num_epochs: int) -> None:
        """
        Train the GAN model for a specified number of epochs.

        Args:
            num_epochs (int): Total number of training epochs.
        """
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch in loop:
                device_batch = self._move_batch_to_device(batch)

                # Use model's custom training_step
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
                
            current_accuracy = self.evaluate()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.save(suffix="best_model")
                print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")

            self.save(suffix="latest")


    def evaluate(self) -> float:
        """
        Evaluate the GAN model on the validation set using PSNR.

        Returns:
            float: Average PSNR score.
        """
        self.model.eval()
        total_psnr = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
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

        avg_psnr = total_psnr / len(self.val_loader)
        print(f"\nValidation PSNR: {avg_psnr:.2f} dB")
        return avg_psnr

    def _move_batch_to_device(
        self, batch: Union[Dict[str, torch.Tensor], list, tuple]
    ) -> Union[Dict[str, torch.Tensor], list[torch.Tensor]]:
        """
        Move batch data to the correct device.

        Args:
            batch: Input batch from the DataLoader.

        Returns:
            Batch moved to the appropriate device.
        """
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        return [x.to(self.device) for x in batch]
