from typing import Any, Dict, Optional, Union, Callable
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp.grad_scaler import GradScaler

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
        super().__init__(model, train_loader, val_loader, device, artifact_dumper=artifact_dumper, **kwargs)

        self.loss_fns = loss_fn
        self.scheduler = scheduler
        self.artifact_dumper = artifact_dumper
        self.log_every = getattr(self.artifact_dumper, "log_every", 10) if self.artifact_dumper else None
        self.global_step = 0
        self.grad_log_interval = kwargs.get("grad_log_interval", 100)
        self.param_log_interval = kwargs.get("param_log_interval", 500)

        if optimizer_args is None:
            optimizer_args = {"lr": 1e-4}

        # ✅ Only build optimizers if optimizer_cls is provided
        if optimizer_cls:
            self.optimizer = {
                key: optimizer_cls[key](self.model.get_submodule(key).parameters(), **optimizer_args)
                for key in ["generator", "discriminator"]
            }
        else:
            self.optimizer = None  # ✅ Needed during evaluation/inference

        
        self.scaler = {
            "generator": GradScaler(enabled=(device == "cuda")),
            "discriminator": GradScaler(enabled=(device == "cuda"))
        }

    def train(self, num_epochs: int) -> Dict[str, float]:
        best_psnr = float("-inf")

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            total_g_loss = 0.0
            total_d_loss = 0.0

            for batch_id, batch in enumerate(loop):
                try:
                    device_batch = self._move_batch_to_device(batch)
                    lr = device_batch["lr"]
                    hr = device_batch["hr"]

                    # === Discriminator ===
                    
                    self.optimizer["discriminator"].zero_grad(set_to_none=True)
                    with torch.autocast(device_type=self.device.type):
                        disc_losses = self.model.training_step(
                            {"lr": lr, "hr": hr},
                            optimizer=self.optimizer,
                            loss_fn=self.loss_fns,
                            device=self.device,
                            phase="discriminator"
                        )
                        d_loss_out = disc_losses["d_loss"]
                        d_loss = d_loss_out.total

                    self.scaler["discriminator"].scale(d_loss).backward()
                    self.scaler["discriminator"].step(self.optimizer["discriminator"])
                    self.scaler["discriminator"].update()

                    # === Generator ===
                    self.optimizer["generator"].zero_grad(set_to_none=True)
                    with torch.autocast(device_type=self.device.type):
                        gen_losses = self.model.training_step(
                            {"lr": lr, "hr": hr},
                            optimizer=self.optimizer,
                            loss_fn=self.loss_fns,
                            device=self.device,
                            phase="generator"
                        )
                        g_loss_out = gen_losses["g_loss"]
                        g_loss = g_loss_out.total


                    self.scaler["generator"].scale(g_loss).backward()
                    self.scaler["generator"].step(self.optimizer["generator"])
                    self.scaler["generator"].update()

                    total_g_loss += g_loss.item()
                    total_d_loss += d_loss.item()

                    loop.set_postfix({
                        "gen_loss": g_loss.item(),
                        "disc_loss": d_loss.item(),
                    })
                    self.global_step += 1

                    # === Logging ===
                    if self.artifact_dumper and self.artifact_dumper.should_log_step(self.global_step):
                        merged_loss = LossOutput(
                            total=g_loss_out.total + d_loss_out.total,
                            components={
                                "generator": g_loss_out.total,
                                "discriminator": d_loss_out.total
                            }
                        )

                        self.artifact_dumper.log_loss(
                            merged_loss,
                            step=self.global_step,
                            prefix="train"
                        )

                        if batch_id % 100 == 0:
                            with torch.no_grad():
                                sr = self.model.generate(lr)
                            model_output = ModelOutput(
                                image=sr,
                                targets=hr,
                                extra={"low_res": lr},
                            )
                            self.artifact_dumper.log_full_output(
                                output=model_output,
                                loss=merged_loss,
                                step=self.global_step,
                                batch_id=batch_id,
                                prefix="train"
                            )

                    # === Logger Hooks ===
                    logger = self._get_logger()
                    if logger:
                        if self.global_step % self.grad_log_interval == 0:
                            logger.log_gradients(self.model.generator, step=self.global_step, prefix="generator")
                            logger.log_gradients(self.model.discriminator, step=self.global_step, prefix="discriminator")
                        if self.global_step % self.param_log_interval == 0:
                            logger.log_parameters(self.model.generator, step=self.global_step, prefix="generator")
                            logger.log_parameters(self.model.discriminator, step=self.global_step, prefix="discriminator")
                            lr_g = self.optimizer["generator"].param_groups[0]["lr"]
                            lr_d = self.optimizer["discriminator"].param_groups[0]["lr"]
                            logger.log_metrics({
                                "lr_generator": lr_g,
                                "lr_discriminator": lr_d
                            }, step=self.global_step)

                except (RuntimeError, ValueError, TypeError) as e:
                    loop.write(f"[ERROR] Batch skipped due to error: {e}")

            if self.scheduler:
                self.scheduler["generator"].step()
                self.scheduler["discriminator"].step()

            avg_psnr = self.evaluate()
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                self.save(suffix="best_model")
                print(f"New best model saved with PSNR: {best_psnr:.2f} dB")

            self.save(suffix="latest")
            avg_g_loss = total_g_loss / len(self.train_loader)
            avg_d_loss = total_d_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

        return {
            "train/g_loss": avg_g_loss,
            "train/d_loss": avg_d_loss,
            "val/psnr": best_psnr
        }


    def evaluate(self) -> float:
        self.model.eval()
        total_psnr = 0.0

        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm(self.val_loader, desc="Evaluating", leave=False)):
                device_batch = self._move_batch_to_device(batch)
                lr = device_batch["lr"]
                hr = device_batch["hr"]
                
                out = self.model.generate(lr)
                sr = out.image if isinstance(out, ModelOutput) else out

                mse = torch.mean((sr - hr) ** 2)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                total_psnr += psnr.item()

                if self.artifact_dumper and self.artifact_dumper.should_log_step(self.global_step):
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
                        prefix="val"
                    )

        avg_psnr = total_psnr / len(self.val_loader)
        print(f"\nValidation PSNR: {avg_psnr:.2f} dB")
        return avg_psnr

    def _move_batch_to_device(
        self, batch: Union[Dict[str, torch.Tensor], list, tuple]
    ) -> Union[Dict[str, torch.Tensor], list[torch.Tensor]]:
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        return [x.to(self.device) for x in batch]
    
    def _get_logger(self):
        return getattr(self.artifact_dumper, "logger", None)
