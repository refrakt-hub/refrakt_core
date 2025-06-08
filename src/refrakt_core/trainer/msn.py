from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.utils.methods import random_patch_masking


@register_trainer("msn")
class MSNTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer_cls,
        optimizer_args=None,
        device="cpu",
        scheduler=None,
        **kwargs,
    ):
        # Add model_name and save_dir via kwargs
        super().__init__(model, train_loader, val_loader, device, **kwargs)

        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.ema_base = kwargs.pop("ema_base", 0.996)
        self.grad_clip = kwargs.pop("grad_clip", None)

        if optimizer_args is None:
            optimizer_args = {}
        self.optimizer = optimizer_cls(model.parameters(), **optimizer_args)

        self.global_step = 0

    def update_ema(self, momentum):
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

    def train(self, num_epochs):
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

            for batch in pbar:
                x = batch[0].to(self.device)

                # Masked (anchor) and unmasked (target) views
                x_anchor = random_patch_masking(x, mask_ratio=0.6, patch_size=16)
                x_target = x

                self.optimizer.zero_grad()
                z_anchor, z_target, prototypes = self.model(x_anchor, x_target)
                loss = self.loss_fn(z_anchor, z_target, prototypes)

                loss.backward()
                if self.grad_clip is not None:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                # EMA update
                momentum = self.ema_base + (1 - self.ema_base) * (
                    self.global_step / 10000
                )
                self.update_ema(momentum)

                self.global_step += 1
                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(self.train_loader)
            print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

    def evaluate(self):
        # Implement basic evaluation if needed
        print(
            "[MSNTrainer] Evaluation not implemented for self-supervised pretraining."
        )
        return 0.0  # Return float for consistency
