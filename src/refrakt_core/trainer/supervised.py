import torch
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer


@register_trainer("supervised")
class SupervisedTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer_cls,
        optimizer_args=None,
        device="cuda",
        scheduler=None,  # Add optional scheduler parameter
        **kwargs,  # Capture additional arguments for future compatibility
    ):
        super().__init__(model, train_loader, val_loader, device, **kwargs)
        self.loss_fn = loss_fn
        if optimizer_args is None:
            optimizer_args = {"lr": 1e-4}

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)
        self.scheduler = scheduler  # Store scheduler if provided
        self.extra_params = kwargs  # Store additional parameters

    def train(self, num_epochs):
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in loop:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                elif isinstance(batch, dict):
                    inputs, targets = batch["input"], batch["target"]
                else:
                    raise TypeError("Unsupported batch format")

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                loop.set_postfix({"loss": loss.item()})

            # Step scheduler at the end of each epoch if available
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch+1} complete. Learning rate: {current_lr:.6f}")

            # Evaluate after each epoch and save best model
            current_accuracy = self.evaluate()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.save(suffix="best_model")
                print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")

            # Always save the latest model
            self.save(suffix="latest")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)
            for batch in loop:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                elif isinstance(batch, dict):
                    inputs, targets = batch["input"], batch["target"]
                else:
                    raise TypeError("Unsupported batch format")

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                loop.set_postfix(
                    {"acc": f"{(correct / total * 100):.2f}%" if total > 0 else "0.00%"}
                )

        accuracy = correct / total if total > 0 else 0
        print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
        return accuracy
