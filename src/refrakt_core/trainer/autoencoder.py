"""
AETrainer implementation for autoencoder-based unsupervised learning tasks.

This module defines the AETrainer class, which handles training and evaluation
of autoencoder models. It supports logging, artifact dumping, and checkpointing.
"""

import os
import random
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.trainer.utils.autoencoder_utils import (
    extract_autoencoder_inputs,
    handle_autoencoder_evaluation_step,
    handle_autoencoder_training_step,
)
from refrakt_core.trainer.utils.string_utils import to_snake_case

T = TypeVar("T", bound=torch.Tensor)


@register_trainer("autoencoder")
class AETrainer(BaseTrainer):
    """
    Trainer for autoencoder-based unsupervised learning tasks.

    Handles training, evaluation, logging, and artifact dumping for autoencoder models.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        loss_fn: Callable[[ModelOutput, torch.Tensor], LossOutput],
        optimizer_cls: Callable[..., Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        artifact_dumper: Optional[Any] = None,
        visualization_hooks: Optional[list] = None,
        explainability_hooks: Optional[list] = None,  # NEW: XAI hooks
        **kwargs: Any,
    ) -> None:
        """
        Initialize the AETrainer.

        Args:
            model (Module): The autoencoder model to be trained.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            loss_fn (Callable): Loss function for autoencoder training.
            optimizer_cls (Callable[..., Optimizer]): Optimizer class.
            optimizer_args (Optional[Dict[str, Any]]): Arguments for the optimizer.
            device (str, optional): Device to use (default: "cuda").
            scheduler (Optional[Any], optional): Learning rate scheduler.
            artifact_dumper (Optional[Any], optional): Artifact logger/dumper.
            **kwargs: Additional keyword arguments.
        """
        # Ensure model_name is properly set from kwargs
        if "model_name" not in kwargs:
            # Fallback to the old behavior if model_name is not provided
            variant = kwargs.pop("model_variant", "simple")
            kwargs["model_name"] = f"autoencoder_{variant}"

        super().__init__(
            model,
            train_loader,
            val_loader,
            device,
            artifact_dumper=artifact_dumper,
            **kwargs,
        )

        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.log_every = (
            getattr(artifact_dumper, "log_every", 1) if artifact_dumper else None
        )  # Changed to 1 for every step
        self.global_step = 0
        self.visualization_hooks = visualization_hooks or []
        self.explainability_hooks = explainability_hooks or []  # NEW: XAI hooks
        self.experiment_id = kwargs.get("experiment_id", None)

        if optimizer_args is None:
            optimizer_args = {"lr": 1e-3}

        self.logger = self._get_logger()
        self.optimizer: Optional[Union[Optimizer, Dict[str, Optimizer]]] = (
            optimizer_cls(self.model.parameters(), **optimizer_args)
        )

    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        Train the autoencoder model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
        """
        start_time = time.time()
        best_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for _step, batch in enumerate(loop):
                inputs = extract_autoencoder_inputs(batch)
                inputs = inputs.to(self.device)

                loss_value, output = handle_autoencoder_training_step(
                    model=self.model,
                    inputs=inputs,
                    loss_fn=self.loss_fn,
                    optimizer=self.optimizer,
                    global_step=self.global_step,
                    artifact_dumper=self.artifact_dumper,
                )

                # --- Visualization hooks: update after each batch ---
                for viz in self.visualization_hooks:
                    try:
                        viz.update_from_batch(self.model, batch, loss_value, epoch)
                    except Exception as e:
                        print(f"[VizHook] update_from_batch() failed: {e}")

                self.global_step += 1
                loop.set_postfix({"loss": loss_value})

            # Validation
            val_loss = self.evaluate()

            if val_loss < best_loss:
                best_loss = val_loss
                self.save(suffix="best_model")
                print(f"New best model saved with loss: {val_loss:.4f}")

            self.save(suffix="latest")

            # --- Visualization hooks: save at end of epoch ---
            model_name = getattr(
                self.model, "model_name", getattr(self, "model_name", "model")
            )
            for viz in self.visualization_hooks:
                registry_name = None
                try:
                    registry_name = getattr(
                        viz, "registry_name", viz.__class__.__name__
                    )
                    print(f"[VizHook] Preparing to save: {registry_name}")
                    # Compose model_name and variant for directory, avoid duplicate variant
                    variant = getattr(
                        self.model,
                        "variant",
                        getattr(self.model, "backbone", None)
                        and getattr(self.model.backbone, "variant", "simple")
                        or "simple",
                    )
                    model_name = getattr(
                        self.model,
                        "model_name",
                        getattr(self, "model_name", "autoencoder"),
                    )
                    if model_name.endswith(f"_{variant}"):
                        model_variant_dir = model_name
                    else:
                        model_variant_dir = f"{model_name}_{variant}"
                    if registry_name == "latent_space_projection":
                        out_dir = f"visualizations/{model_variant_dir}"
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = f"{out_dir}/{registry_name}.png"
                        print(
                            f"[VizHook] Calling save for {registry_name} with path {out_path} (with epoch)"
                        )
                        viz.save(out_path, epoch=epoch + 1)
                    elif registry_name in [
                        "disentanglement_analysis",
                        "feature_attribution",
                        "reconstruction_viz",
                        "sample_generation",
                    ]:
                        out_dir = f"visualizations/{model_variant_dir}"
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = f"{out_dir}/{registry_name}.png"
                        print(
                            f"[VizHook] Calling save for {registry_name} with path {out_path}"
                        )
                        viz.save(out_path)
                except Exception as e:
                    if registry_name is not None:
                        print(f"[VizHook] save() failed for {registry_name}: {e}")
                    else:
                        print(f"[VizHook] save() failed: {e}")

            # --- Explainability hooks: run at end of epoch ---
            self._run_explainability_hooks(epoch, inference=False)

        training_time = time.time() - start_time
        return {
            "final_loss": best_loss,
            "best_loss": best_loss,
            "training_time": training_time,
            "epochs_completed": num_epochs,
        }

    def evaluate(self) -> float:
        """
        Evaluate the autoencoder model on the validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)

            for val_step, batch in enumerate(loop):
                # Use a separate step counter for validation to avoid conflicts
                val_global_step = (
                    self.global_step + val_step + 1000000
                )  # Large offset to avoid conflicts

                inputs = extract_autoencoder_inputs(batch)
                inputs = inputs.to(self.device)

                loss_value = handle_autoencoder_evaluation_step(
                    model=self.model,
                    inputs=inputs,
                    loss_fn=self.loss_fn,
                    global_step=val_global_step,
                    artifact_dumper=self.artifact_dumper,
                )

                total_loss += loss_value

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def _run_explainability_hooks(self, epoch: int, inference: bool = False) -> None:
        """
        Run all explainability hooks on a small batch from the validation set and save attributions as images or arrays.
        Args:
            epoch (int): Current epoch number (for output directory naming)
            inference (bool): If True, only save a single random sample per batch (for explanations_inference)
        """
        if not self.explainability_hooks:
            return
        if not hasattr(self, "val_loader") or self.val_loader is None:
            return
        try:
            sample_batch = next(iter(self.val_loader))
            if isinstance(sample_batch, (tuple, list)):
                input_tensor = sample_batch[0]
            elif isinstance(sample_batch, dict):
                input_tensor = sample_batch.get("image") or sample_batch.get("input")
                if input_tensor is None:
                    raise ValueError(
                        "Batch dict does not contain a valid 'image' or 'input' tensor for XAI."
                    )
            else:
                input_tensor = sample_batch
            input_tensor = input_tensor.to(self.device)
        except Exception as e:
            print(f"[XAI] Could not get validation batch for explainability: {e}")
            return
        for xai_cls, params in self.explainability_hooks:
            try:
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
                        self, "model_name", "autoencoder"
                    )
                    if hasattr(self, "experiment_id") and self.experiment_id:
                        experiment_id = self.experiment_id
                    else:
                        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Use checkpoints directory structure
                    checkpoints_base_dir = f"./checkpoints/{model_name}_{experiment_id}"
                    save_runtime_xai_info(
                        xai_method,
                        xai_cls.__name__,
                        params,
                        checkpoints_base_dir,
                        logger=None,
                    )
                except Exception as e:
                    pass  # Silently ignore to avoid breaking XAI execution

                attributions = xai_method.explain(input_tensor)
                # Save attribution as image (if 2D/3D) or numpy array
                import os

                import matplotlib.pyplot as plt
                import numpy as np

                # Use registry_name from params if present, else method name, always lowercased with underscores, no 'xai' suffix
                registry_name = params.get(
                    "registry_name", params.get("method", xai_cls.__name__)
                ).replace(" ", "_")
                if registry_name.lower().endswith("xai"):
                    registry_name = registry_name[:-3]
                registry_name = to_snake_case(registry_name)

                # Get model name and datetime for unique output dir (same as supervised trainer)
                model_name = getattr(self.model, "model_name", None) or getattr(
                    self, "model_name", "autoencoder"
                )
                # Use experiment_id if available, otherwise generate timestamp
                if hasattr(self, "experiment_id") and self.experiment_id:
                    dt_str = self.experiment_id
                else:
                    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Use unified explanations directory structure
                base_dir = os.path.join(
                    "./explanations", f"{model_name}_{dt_str}", "train", registry_name
                )
                os.makedirs(base_dir, exist_ok=True)

                arr = attributions.detach().cpu().numpy()
                # For explanations_inference, only save a single random sample (regardless of shape)
                if inference and arr.shape[0] > 1:
                    idx = random.randint(0, arr.shape[0] - 1)
                    arr = arr[idx]
                # If arr is 4D (B, C, H, W), or 3D (B, H, W), ensure only one sample is saved
                elif not inference and arr.ndim >= 3 and arr.shape[0] > 1:
                    arr = arr[0]
                fname = f"{base_dir}/{registry_name}.png"
                # Always save NPY file for numerical analysis
                npy_fname = f"{base_dir}/{registry_name}.npy"
                np.save(npy_fname, arr)

                if arr.ndim in [2, 3]:
                    plt.figure()
                    if arr.ndim == 3 and arr.shape[0] in [1, 3]:  # e.g., (C, H, W)
                        # If single-channel or RGB, show as image
                        arr_disp = (
                            arr.transpose(1, 2, 0) if arr.shape[0] == 3 else arr[0]
                        )
                        plt.imshow(arr_disp, cmap="hot")
                    else:
                        plt.imshow(arr, cmap="hot")
                    plt.colorbar()
                    plt.title(f"{registry_name} Attribution")
                    plt.savefig(fname)
                    plt.close()
                else:
                    # For non-image attributions, just save NPY (already done above)
                    pass
                print(f"[XAI] Saved attribution: {fname}")
            except Exception as e:
                print(f"[XAI] {xai_cls.__name__} failed: {e}")

    def _run_explainability_inference(self):
        """
        Run all explainability hooks on a single random sample from the validation set and save attributions as images.
        """
        if not self.explainability_hooks:
            return
        if not hasattr(self, "val_loader") or self.val_loader is None:
            return
        try:
            sample_batch = next(iter(self.val_loader))
            if isinstance(sample_batch, (tuple, list)):
                input_tensor = sample_batch[0]
            elif isinstance(sample_batch, dict):
                input_tensor = sample_batch.get("image") or sample_batch.get("input")
                if input_tensor is None:
                    raise ValueError(
                        "Batch dict does not contain a valid 'image' or 'input' tensor for XAI."
                    )
            else:
                input_tensor = sample_batch
            input_tensor = input_tensor.to(self.device)
        except Exception as e:
            print(f"[XAI] Could not get validation batch for explainability: {e}")
            return
        for xai_cls, params in self.explainability_hooks:
            try:
                if xai_cls.__name__ == "ConceptSaliencyXAI":
                    xai_method = xai_cls(
                        self.model,
                        dataloader=self.val_loader,
                        device=self.device,
                        **params,
                    )
                else:
                    xai_method = xai_cls(self.model, **params)
                attributions = xai_method.explain(input_tensor)
                arr = attributions.detach().cpu().numpy()
                # Always select a single random sample, regardless of shape
                if arr.shape[0] > 1:
                    idx = random.randint(0, arr.shape[0] - 1)
                    arr = arr[idx]
                # If arr is still 4D (C, H, W, ...), reduce to 2D or 3D for visualization
                if arr.ndim == 4:
                    arr = arr[0]
                model_name = getattr(
                    self.model, "model_name", getattr(self, "model_name", "autoencoder")
                )
                registry_name = params.get(
                    "registry_name", params.get("method", xai_cls.__name__)
                ).replace(" ", "_")
                if registry_name.lower().endswith("xai"):
                    registry_name = registry_name[:-3]
                registry_name = to_snake_case(registry_name)

                # Get model name and datetime for unique output dir (same as supervised trainer)
                # Use experiment_id if available, otherwise generate timestamp
                if hasattr(self, "experiment_id") and self.experiment_id:
                    dt_str = self.experiment_id
                else:
                    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Use unified explanations directory structure
                base_dir = os.path.join(
                    "./explanations",
                    f"{model_name}_{dt_str}",
                    "inference",
                    registry_name,
                )
                os.makedirs(base_dir, exist_ok=True)

                fname = f"{base_dir}/{registry_name}.png"
                # Always save NPY file for numerical analysis
                npy_fname = f"{base_dir}/{registry_name}.npy"
                np.save(npy_fname, arr)

                # Save as image
                plt.figure()
                if arr.ndim == 3 and arr.shape[0] in [1, 3]:
                    arr_disp = arr.transpose(1, 2, 0) if arr.shape[0] == 3 else arr[0]
                    plt.imshow(arr_disp, cmap="hot")
                else:
                    plt.imshow(arr, cmap="hot")
                plt.colorbar()
                plt.title(f"{registry_name} Attribution")
                plt.savefig(fname)
                plt.close()
                print(f"[XAI] Saved inference attribution: {fname}")
            except Exception as e:
                print(f"[XAI] {xai_cls.__name__} failed: {e}")

    def _unwrap_output(
        self, output: Union[ModelOutput, Dict[str, Any], torch.Tensor]
    ) -> ModelOutput:
        """
        Convert output to ModelOutput if not already.

        Args:
            output (Union[ModelOutput, Dict[str, Any], torch.Tensor]): Model output.

        Returns:
            ModelOutput: Wrapped model output.

        Raises:
            ValueError: If output is None.
        """
        if output is None:
            raise ValueError("[_unwrap_output] Received None as output!")

        if isinstance(output, ModelOutput):
            return output
        elif isinstance(output, dict):
            return ModelOutput(**output)
        else:
            return ModelOutput(reconstruction=output)

    def _extract_inputs(
        self, batch: Union[torch.Tensor, Dict[str, Any], list[Any], tuple[Any, ...]]
    ) -> torch.Tensor:
        """
        Extract input tensor from a batch.

        Args:
            batch (Union[torch.Tensor, Dict[str, Any], list, tuple]): Batch from DataLoader.

        Returns:
            torch.Tensor: Input tensor.

        Raises:
            TypeError: If input tensor cannot be extracted.
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) == 0 or not isinstance(batch[0], torch.Tensor):
                raise TypeError(
                    "Batch is empty or does not contain a tensor as the first element."
                )
            return batch[0]
        if isinstance(batch, dict):
            image = batch.get("image")
            if image is not None and isinstance(image, torch.Tensor):
                return cast(torch.Tensor, image)
            input_tensor = batch.get("input")
            if input_tensor is not None and isinstance(input_tensor, torch.Tensor):
                return cast(torch.Tensor, input_tensor)
            raise TypeError(
                "Batch dict does not contain a valid 'image' or 'input' tensor."
            )
        if isinstance(batch, torch.Tensor):
            return batch
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _get_logger(self) -> Optional[Any]:
        """
        Retrieve the logger from the artifact dumper if available.

        Returns:
            Optional[Any]: Logger object if available, else None.
        """
        if self.artifact_dumper and hasattr(self.artifact_dumper, "logger"):
            return self.artifact_dumper.logger
        return None
