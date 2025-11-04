"""
GANTrainer implementation for Generative Adversarial Network training tasks.

This module defines the GANTrainer class, which handles training and evaluation
of GAN models (e.g., SRGAN, StyleGAN, etc.). It supports logging, artifact dumping,
and checkpointing for both generator and discriminator components.
"""

import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
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
        visualization_hooks: Optional[list] = None,
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
            # Handle both optimizer classes and optimizer instances
            if isinstance(optimizer_cls, dict):
                # Check if we have optimizer instances or classes
                first_key = next(iter(optimizer_cls.keys()))
                first_optimizer = optimizer_cls[first_key]

                if isinstance(first_optimizer, Optimizer):
                    # We already have optimizer instances, use them directly
                    self.optimizer: Optional[Union[Optimizer, Dict[str, Optimizer]]] = optimizer_cls  # type: ignore
                else:
                    # We have optimizer classes, instantiate them
                    self.optimizer = {
                        key: optimizer_cls[key](
                            self.model.get_submodule(key).parameters(), **optimizer_args
                        )
                        for key in ["generator", "discriminator"]
                    }
            else:
                # Handle single optimizer case (shouldn't happen for GAN but for safety)
                self.optimizer = optimizer_cls
        else:
            self.optimizer = None

        self.scaler = {
            "generator": GradScaler(enabled=(device == "cuda")),
            "discriminator": GradScaler(enabled=(device == "cuda")),
        }

        self.visualization_hooks = visualization_hooks or []
        self.explainability_hooks = kwargs.get("explainability_hooks", [])

    def _run_explainability_hooks(self, epoch: int, inference: bool = False):
        """Run explainability hooks at the end of an epoch."""
        if not self.explainability_hooks:
            return
        if not hasattr(self, "val_loader") or self.val_loader is None:
            return

        try:
            # Get a sample from validation loader for XAI analysis
            sample_batch = next(iter(self.val_loader))
            if isinstance(sample_batch, dict):
                # For SRGAN, inputs are typically {"lr": low_res, "hr": high_res}
                input_tensor = sample_batch.get("lr", sample_batch.get("input"))
                if input_tensor is None:
                    raise ValueError(
                        "Batch dict does not contain a valid 'lr' or 'input' tensor for XAI."
                    )
            elif isinstance(sample_batch, (list, tuple)):
                input_tensor = sample_batch[0]  # First element is typically input
            else:
                input_tensor = sample_batch

            input_tensor = input_tensor.to(self.device)
        except Exception as e:
            logger = self._get_logger()
            if logger:
                logger.warning(
                    f"[XAI] Could not get validation batch for explainability: {e}"
                )
            else:
                print(f"[XAI] Could not get validation batch for explainability: {e}")
            return

        for xai_cls, params in self.explainability_hooks:
            try:
                # Instantiate XAI method like other trainers
                if xai_cls.__name__ == "ConceptSaliencyXAI":
                    xai_method = xai_cls(
                        self.model,
                        dataloader=self.val_loader,
                        device=self.device,
                        **params,
                    )
                else:
                    xai_method = xai_cls(self.model, **params)

                # Save runtime XAI info for metadata collection
                try:
                    from refrakt_cli.helpers.shared_core import save_runtime_xai_info

                    # Determine base directory for saving runtime info
                    model_name = getattr(self.model, "model_name", None) or getattr(
                        self, "model_name", "srgan"
                    )
                    experiment_id = getattr(self, "experiment_id", None)
                    if not experiment_id:
                        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Use checkpoints directory structure
                    checkpoints_base_dir = f"./checkpoints/{model_name}_{experiment_id}"
                    save_runtime_xai_info(
                        xai_method,
                        xai_cls.__name__,
                        params,
                        checkpoints_base_dir,
                        self._get_logger(),
                    )
                except Exception as e:
                    logger = self._get_logger()
                    if logger:
                        logger.warning(f"Failed to save runtime XAI info: {e}")

                # Generate explanations using the standard interface
                attributions = xai_method.explain(input_tensor)

                # Save attributions like the other trainers
                # Use registry_name from params if present, else method name, always lowercased with underscores, no 'xai' suffix
                registry_name = params.get(
                    "registry_name", params.get("method", xai_cls.__name__)
                ).replace(" ", "_")
                if registry_name.lower().endswith("xai"):
                    registry_name = registry_name[:-3]

                # Use to_snake_case for consistent naming with other trainers
                def to_snake_case(name):
                    """Convert camelCase or PascalCase to snake_case."""
                    import re

                    # Handle special cases
                    if name == "LayerGradCAM":
                        return "layer_gradcam"
                    elif name == "GradCAM":
                        return "gradcam"
                    elif name == "IntegratedGradients":
                        return "integrated_gradients"

                    # General conversion
                    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
                    name = name.replace("__", "_")
                    return name.strip("_")

                registry_name = to_snake_case(registry_name)

                # Get model name and datetime for unique output dir
                model_name = getattr(self.model, "model_name", None) or getattr(
                    self, "model_name", "srgan"
                )
                # Use experiment_id if available, otherwise generate timestamp
                dt_str = getattr(self, "experiment_id", None)
                if not dt_str:
                    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Use explanations directory structure for training
                phase = "inference" if inference else "train"
                base_dir = os.path.join(
                    "./explanations", f"{model_name}_{dt_str}", phase, registry_name
                )
                os.makedirs(base_dir, exist_ok=True)

                # Convert attributions to numpy and save each sample
                attr_np = attributions.detach().cpu().numpy()

                for i in range(min(attr_np.shape[0], 8)):
                    arr = attr_np[i]

                    # Handle different attribution shapes
                    if len(arr.shape) == 0:  # Scalar
                        continue
                    elif (
                        len(arr.shape) == 1 and arr.shape[0] == 1
                    ):  # [1] - class-specific attribution
                        continue
                    elif len(arr.shape) == 1:  # 1D array
                        # Reshape to 2D for visualization
                        arr = arr.reshape(
                            (int(np.sqrt(arr.shape[0])), int(np.sqrt(arr.shape[0])))
                        )
                    elif len(arr.shape) == 2:  # 2D array (spatial heatmap)
                        pass  # Keep as is
                    elif len(arr.shape) == 3:  # 3D array
                        if arr.shape[0] == 1:  # [1, H, W]
                            arr = arr[0]
                        elif arr.shape[0] == 3:  # [3, H, W] - RGB
                            arr = np.transpose(arr, (1, 2, 0))
                        else:
                            continue
                    else:
                        continue

                    # Normalize and convert to image
                    arr = arr - arr.min()
                    arr = arr / (arr.max() + 1e-8)
                    arr = (arr * 255).astype(np.uint8)

                    # Convert to PIL Image
                    if len(arr.shape) == 2:  # Grayscale
                        from PIL import Image

                        img = Image.fromarray(arr, mode="L")
                    elif len(arr.shape) == 3:  # RGB
                        from PIL import Image

                        img = Image.fromarray(arr, mode="RGB")
                    else:
                        continue

                    img_path = os.path.join(base_dir, f"sample_{i}.png")
                    img.save(img_path)

                    # Save raw attribution as .npy
                    npy_path = os.path.join(base_dir, f"sample_{i}.npy")
                    np.save(npy_path, attr_np[i])

                    import json

                    metadata = {
                        "method": registry_name,
                        "epoch": epoch + 1,
                        "sample_index": i,
                        "image_path": img_path,
                        "attribution_path": npy_path,
                    }
                    meta_path = os.path.join(base_dir, f"sample_{i}_metadata.json")
                    with open(meta_path, "w") as f:
                        json.dump(metadata, f)

            except Exception as e:
                logger = self._get_logger()
                if logger:
                    logger.warning(
                        f"[XAI] Failed to run or save explainability for {xai_cls}: {e}"
                    )
                else:
                    print(
                        f"[XAI] Failed to run or save explainability for {xai_cls}: {e}"
                    )

    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        Train the GAN model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
        """
        import time

        start_time = time.time()
        best_psnr = float("-inf")
        final_avg_g_loss = 0.0
        final_avg_d_loss = 0.0

        for epoch in range(num_epochs):

            # Create progress bar but don't consume the iterator
            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
            )

            # Train for one epoch
            total_g_loss, total_d_loss = handle_gan_epoch_training(
                model=self.model,
                train_loader=progress_bar,  # Use progress bar iterator
                optimizer=self.optimizer,
                loss_fns=self.loss_fns,
                device=self.device,
                scaler=self.scaler,
                global_step=self.global_step,
                artifact_dumper=self.artifact_dumper,
                grad_log_interval=self.grad_log_interval,
                param_log_interval=self.param_log_interval,
                logger=self._get_logger(),
                visualization_hooks=self.visualization_hooks,
            )
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

            # --- Visualization hooks: save at end of epoch ---
            model_name = getattr(
                self.model, "model_name", getattr(self, "model_name", "model")
            )
            for viz in self.visualization_hooks:
                try:
                    registry_name = getattr(
                        viz, "registry_name", viz.__class__.__name__
                    )
                    out_dir = f"visualizations/{model_name}/{registry_name}"
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = f"{out_dir}/epoch_{epoch+1}.png"
                    viz.save(out_path)
                except Exception as e:
                    print(f"[VizHook] save() failed: {e}")

            # --- Explainability hooks: run at end of epoch ---
            self._run_explainability_hooks(epoch, inference=False)

        training_time = time.time() - start_time
        return {
            "final_loss": final_avg_g_loss,  # Use generator loss as primary loss
            "final_g_loss": final_avg_g_loss,
            "final_d_loss": final_avg_d_loss,
            "best_psnr": best_psnr,
            "best_accuracy": best_psnr,  # Use PSNR as accuracy proxy for compatibility
            "training_time": training_time,
            "epochs_completed": num_epochs,
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
