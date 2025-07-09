"""
GANTrainer implementation for Generative Adversarial Network training tasks.

This module defines the GANTrainer class, which handles training and evaluation
of GAN models (e.g., SRGAN, StyleGAN, etc.). It supports logging, artifact dumping,
and checkpointing for both generator and discriminator components.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.amp.grad_scaler import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.trainer.utils.gan_utils import (
    handle_gan_epoch_training,
    handle_gan_scheduler_step,
)


@register_trainer("gan")
class GANTrainer(BaseTrainer):
    """
    Trainer for Generative Adversarial Network (GAN) training tasks.

    Handles training, evaluation, logging, and artifact dumping for GAN models.
    Manages separate optimizers and loss functions for generator and discriminator.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        loss_fn: Dict[str, Callable[..., Any]],
        optimizer_cls: Dict[str, Callable[..., Optimizer]],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        artifact_dumper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the GANTrainer.

        Args:
            model (Module): The GAN model to be trained.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            loss_fn (Dict[str, Callable]): Dictionary of loss functions for generator and discriminator.
            optimizer_cls (Dict[str, Callable[..., Optimizer]]): Dictionary of optimizer classes.
            optimizer_args (Optional[Dict[str, Any]], optional): Arguments for the optimizers.
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
            **kwargs,
        )

        self.loss_fns = loss_fn
        self.scheduler = scheduler
        self.artifact_dumper = artifact_dumper
        self.log_every = (
            getattr(self.artifact_dumper, "log_every", 10)
            if self.artifact_dumper
            else None
        )
        self.global_step = 0
        self.grad_log_interval = kwargs.get("grad_log_interval", 100)
        self.param_log_interval = kwargs.get("param_log_interval", 500)

        if optimizer_args is None:
            optimizer_args = {"lr": 1e-4}

        if optimizer_cls:
            self.optimizer: Optional[Union[Optimizer, Dict[str, Optimizer]]] = {
                key: optimizer_cls[key](
                    self.model.get_submodule(key).parameters(), **optimizer_args
                )
                for key in ["generator", "discriminator"]
            }
        else:
            self.optimizer = None

        self.scaler = {
            "generator": GradScaler(enabled=(device == "cuda")),
            "discriminator": GradScaler(enabled=(device == "cuda")),
        }

    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        Train the GAN model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
        """
        best_psnr = float("-inf")
        final_avg_g_loss = 0.0
        final_avg_d_loss = 0.0

        for epoch in range(num_epochs):
            tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            # Train for one epoch
            total_g_loss, total_d_loss = handle_gan_epoch_training(
                model=self.model,
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                loss_fns=self.loss_fns,
                device=self.device,
                scaler=self.scaler,
                global_step=self.global_step,
                artifact_dumper=self.artifact_dumper,
                grad_log_interval=self.grad_log_interval,
                param_log_interval=self.param_log_interval,
                logger=self._get_logger(),
            )

            # Update schedulers
            handle_gan_scheduler_step(self.scheduler)

            # Evaluate and save
            avg_psnr = self.evaluate()
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                self.save(suffix="best_model")
                print(f"New best model saved with PSNR: {best_psnr:.2f} dB")

            self.save(suffix="latest")
            final_avg_g_loss = (
                total_g_loss / len(self.train_loader)
                if len(self.train_loader) > 0
                else 0.0
            )
            final_avg_d_loss = (
                total_d_loss / len(self.train_loader)
                if len(self.train_loader) > 0
                else 0.0
            )
            print(
                f"Epoch [{epoch+1}/{num_epochs}], G Loss: {final_avg_g_loss:.4f}, D Loss: {final_avg_d_loss:.4f}"
            )

        return {
            "final_g_loss": final_avg_g_loss,
            "final_d_loss": final_avg_d_loss,
            "best_psnr": best_psnr,
        }

    def evaluate(self) -> float:
        """
        Evaluate the GAN model on the validation set.

        Returns:
            float: Average PSNR on validation set.
        """
        self.model.eval()
        total_psnr = 0.0

        with torch.no_grad():
            for batch_id, batch in enumerate(
                tqdm(self.val_loader, desc="Evaluating", leave=False)
            ):
                device_batch = self._move_batch_to_device(batch)

                if isinstance(device_batch, dict):
                    lr = device_batch["lr"]
                    hr = device_batch["hr"]
                elif isinstance(device_batch, (list, tuple)) and len(device_batch) >= 2:
                    lr = device_batch[0]
                    hr = device_batch[1]
                else:
                    raise ValueError(
                        "Batch must be a dict with 'lr' and 'hr' keys or a list/tuple with at least 2 elements"
                    )

                out = self.model.generate(lr)
                sr = out.image if isinstance(out, ModelOutput) else out

                mse = torch.mean((sr - hr) ** 2)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                total_psnr += psnr.item()

                if self.artifact_dumper and self.artifact_dumper.should_log_step(
                    self.global_step
                ):
                    model_output = ModelOutput(
                        image=sr,
                        targets=hr,
                        extra={"low_res": lr},
                    )
                    self.artifact_dumper.log_full_output(
                        output=model_output,
                        loss=None,
                        step=self.global_step,
                        batch_id=batch_id,
                        prefix="val",
                    )

        avg_psnr = total_psnr / len(self.val_loader)
        print(f"\nValidation PSNR: {avg_psnr:.2f} dB")
        return avg_psnr

    def _move_batch_to_device(
        self,
        batch: Union[
            Dict[str, torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor, ...]
        ],
    ) -> Union[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Move batch tensors to the specified device.

        Args:
            batch (Union[Dict[str, torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor, ...]]):
                Batch to move to device.

        Returns:
            Union[Dict[str, torch.Tensor], List[torch.Tensor]]: Batch moved to device.
        """
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        return [x.to(self.device) for x in batch]

    def _get_logger(self) -> Optional[Any]:
        """
        Retrieve the logger from the artifact dumper if available.

        Returns:
            Optional[Any]: Logger object if available, else None.
        """
        return getattr(self.artifact_dumper, "logger", None)
