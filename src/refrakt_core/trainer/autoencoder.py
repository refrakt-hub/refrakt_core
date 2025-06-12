"""
Trainer module for autoencoder models.

This trainer is responsible for handling the training and validation logic
of autoencoder-based models using PyTorch.
"""

from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer


@register_trainer("autoencoder")
class AETrainer(BaseTrainer):
    """
    Autoencoder Trainer.

    Handles training and evaluation loops for autoencoder models.

    Args:
        model (Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        loss_fn (Callable): Loss function.
        optimizer_cls (Callable[..., Optimizer]): Optimizer class (e.g., torch.optim.Adam).
        optimizer_args (Optional[Dict[str, Any]]): Arguments for optimizer instantiation.
        device (str): Device to use ("cuda" or "cpu").
        scheduler (Optional[Any]): Learning rate scheduler.
        **kwargs: Additional arguments forwarded to BaseTrainer.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer_cls: Callable[..., Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        variant = kwargs.pop("model_variant", "simple")
        kwargs["model_name"] = f"autoencoder_{variant}"
        super().__init__(model, train_loader, val_loader, device, **kwargs)
        
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        if optimizer_args is None:
            optimizer_args = {"lr": 1e-3}

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)

    def train(self, num_epochs: int) -> None:
        """
        Train the autoencoder model.

        Args:
            num_epochs (int): Number of epochs to train for.
        """
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch in loop:
                inputs = self._extract_inputs(batch)
                inputs = inputs.to(self.device)

                self.optimizer.zero_grad()
                raw_outputs = self.model(inputs)
                outputs = self._unwrap_output(raw_outputs)
                # if isinstance(raw_outputs, dict):
                #     raw_outputs = raw_outputs["recon"]
                loss = self.loss_fn(outputs, inputs)
                loss.backward()
                self.optimizer.step()

                loop.set_postfix({"loss": loss.item()})

            # Step scheduler if provided
            if self.scheduler:
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch + 1} complete. Learning rate: {lr:.6f}")

            # Evaluate and save checkpoints
            current_loss = self.evaluate()
            if current_loss < best_loss:
                best_loss = current_loss
                self.save(suffix="best_model")
                print(f"New best model saved with loss: {best_loss:.4f}")

            self.save(suffix="latest")

    def evaluate(self) -> float:
        """
        Evaluate the model on the validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)
            for batch in loop:
                inputs = self._extract_inputs(batch)
                inputs = inputs.to(self.device)

                
                raw_outputs = self.model(inputs)
                outputs = self._unwrap_output(raw_outputs)
                # if isinstance(raw_outputs, dict):
                #     raw_outputs = raw_outputs["recon"]
                loss = self.loss_fn(outputs, inputs)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
    
    def _unwrap_output(self, output):
        """
        Unwrap model output for loss computation.
        
        For MAE models, return the full dictionary.
        For other autoencoder models, extract the reconstruction tensor.
        """
        if isinstance(output, dict):
            # Check if this is MAE output (has mask and original_patches)
            if "mask" in output and "original_patches" in output:
                return output  # Return full dictionary for MAE
            elif "recon" in output:
                return output["recon"]
            elif "output" in output:
                return output["output"]
            else:
                raise KeyError("Expected 'recon' or 'output' key in model output.")
        return output


    def _extract_inputs(self, batch: Union[torch.Tensor, Dict, list, tuple]) -> torch.Tensor:
        """
        Extracts the input tensor from a given batch.

        Args:
            batch: Batch of data from DataLoader.

        Returns:
            torch.Tensor: Input tensor.
        """
        if isinstance(batch, (list, tuple)):
            return batch[0]
        if isinstance(batch, dict):
            return batch["image"]
        return batch
    
    def generate_latent_projection(self, dataloader, device, save_path=None, method="pca"):
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        model = self.model
        model.eval()
        all_mu, all_labels = [], []

        for batch in dataloader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else None

            with torch.no_grad():
                output = model(inputs)
                if isinstance(output, dict) and "mu" in output:
                    all_mu.append(output["mu"].cpu())
                    if labels is not None:
                        all_labels.append(labels.cpu())

        mu_tensor = torch.cat(all_mu, dim=0)
        label_tensor = torch.cat(all_labels, dim=0) if all_labels else None
        mu_np = mu_tensor.numpy()

        if mu_np.shape[1] > 2:
            reducer = PCA(n_components=2) if method == "pca" else TSNE(n_components=2)
            mu_2d = reducer.fit_transform(mu_np)
        else:
            mu_2d = mu_np

        plt.figure(figsize=(8, 6))
        if label_tensor is not None:
            plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=label_tensor.numpy(), cmap="tab10", alpha=0.7)
            plt.colorbar()
        else:
            plt.scatter(mu_2d[:, 0], mu_2d[:, 1], alpha=0.7)

        plt.title(f"Latent Space Projection ({method.upper()})")
        plt.xlabel("z1")
        plt.ylabel("z2")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def generate_latent_grid_samples(self, grid_size=20, range_lim=3.0):
        assert self.model.hidden_dim == 2, "Only for 2D latent space"
        import matplotlib.pyplot as plt
        
        z = torch.stack([
            torch.tensor([x, y]) for y in torch.linspace(-range_lim, range_lim, grid_size)
                                for x in torch.linspace(-range_lim, range_lim, grid_size)
        ])
        z = z.to(self.device)
        with torch.no_grad():
            samples = self.model.decode(z).cpu()

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        for i in range(grid_size):
            for j in range(grid_size):
                axes[i, j].imshow(samples[i * grid_size + j].reshape(28, 28), cmap="gray")
                axes[i, j].axis("off")
        plt.suptitle("Latent Traversal Grid")
        plt.show()
