from typing import Any, Callable, Dict, Optional

import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.nn.functional import cosine_similarity

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.utils.methods import random_patch_masking
from refrakt_core.schema.model_output import ModelOutput


@register_trainer("msn")
class MSNTrainer(BaseTrainer):
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer_cls: Callable[..., Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        scheduler: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, train_loader, val_loader, device, **kwargs)

        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.ema_base: float = kwargs.pop("ema_base", 0.996)
        self.grad_clip: Optional[float] = kwargs.pop("grad_clip", None)

        if optimizer_args is None:
            optimizer_args = {"lr": 1e-4}

        # Convert DictConfig to regular dict if needed
        from omegaconf import DictConfig
        if isinstance(optimizer_args, DictConfig):
            from omegaconf import OmegaConf
            optimizer_args = OmegaConf.to_container(optimizer_args, resolve=True)

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)
        self.global_step = 0

    def update_ema(self, momentum: float) -> None:
        for param, ema_param in zip(
            self.model.encoder.parameters(),
            self.model.target_encoder.parameters(),
            strict=False,
        ):
            ema_param.data.mul_(momentum).add_((1 - momentum) * param.data)

        for param, ema_param in zip(
            self.model.projector.parameters(),
            self.model.target_projector.parameters(),
            strict=False,
        ):
            ema_param.data.mul_(momentum).add_((1 - momentum) * param.data)

    def train(self, num_epochs: int) -> None:
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

            for batch in pbar:
                x = batch[0].to(self.device)

                x_anchor = random_patch_masking(x, mask_ratio=0.6, patch_size=16)
                x_target = x

                self.optimizer.zero_grad()

                output: ModelOutput = self.model(x_anchor, x_target)
                z_anchor = output.embeddings
                z_target = output.loss_components.get("z_target")
                prototypes = output.loss_components.get("prototypes")

                loss = self.loss_fn(z_anchor, z_target, prototypes)

                loss.backward()
                if self.grad_clip is not None:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                momentum = self.ema_base + (1 - self.ema_base) * (
                    self.global_step / 10000
                )
                self.update_ema(momentum)
                self.global_step += 1

                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_loss = running_loss / len(self.train_loader)
            print(f"[Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}")
        
    def evaluate(self) -> float:
        """
        Evaluate MSN model by measuring average cosine similarity between
        anchor and target embeddings across the validation set.

        Returns:
            float: Average cosine similarity (0.0 to 1.0)
        """
        self.model.eval()
        total_cos_sim = 0.0
        num_samples = 0

        if self.val_loader is None:
            print("[MSNTrainer] No validation loader provided. Skipping evaluation.")
            return 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch[0].to(self.device)
                x_anchor = x  # use masked view here if you want, e.g., random_patch_masking(x, ...)
                x_target = x

                output: ModelOutput = self.model(x_anchor, x_target)
                z_anchor = output.embeddings  # This is a single tensor
                z_target = output.loss_components.get("z_target")

                if z_anchor is None or z_target is None:
                    continue

                cos_sim = cosine_similarity(z_anchor, z_target, dim=-1)  # shape: (batch_size,)
                total_cos_sim += cos_sim.sum().item()
                num_samples += cos_sim.size(0)

        avg_cos_sim = total_cos_sim / num_samples if num_samples > 0 else 0.0
        print(f"[MSNTrainer] Evaluation - Avg Cosine Similarity: {avg_cos_sim:.4f}")
        return avg_cos_sim
