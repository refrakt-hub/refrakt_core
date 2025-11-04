"""
SupervisedTrainer implementation for standard supervised learning tasks.

This module defines the SupervisedTrainer class, which handles training and evaluation
of models using supervised objectives (e.g., classification, regression).
It supports logging, artifact dumping, and integration with explainability/visualization tools.
"""

import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.trainer.utils.supervised_utils import (
    handle_epoch_end,
    handle_training_step,
    log_artifacts,
    log_training_metrics,
)
from refrakt_core.wrappers.utils.default_loss_utils import (
    extract_tensor_from_model_output,
)
from refrakt_viz.supervised.confusion_matrix import ConfusionMatrixPlot
from refrakt_viz.supervised.loss_accuracy import LossAccuracyPlot
from refrakt_viz.supervised.sample_predictions import SamplePredictionsPlot


@register_trainer("supervised")
class SupervisedTrainer(BaseTrainer):
    """
    Supervised trainer for classification and regression tasks.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        loss_fn: Callable[..., Any],
        optimizer_cls: Callable[..., Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        artifact_dumper: Optional[Any] = None,
        visualization_hooks: Optional[List[Any]] = None,
        explainability_hooks: Optional[List[Any]] = None,
        experiment_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            device=device,
            scheduler=scheduler,
            artifact_dumper=artifact_dumper,
            **kwargs,
        )

        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.experiment_id = experiment_id
        self.extra_params = kwargs
        self.grad_log_interval = kwargs.get("grad_log_interval", 100)
        self.param_log_interval = kwargs.get("param_log_interval", 500)
        self.log_every = (
            getattr(self.artifact_dumper, "log_every", 10)
            if self.artifact_dumper
            else None
        )
        self.global_step = 0
        self._current_batch = None
        self._current_loss_output = None

        self.visualization_hooks = visualization_hooks or []
        self.explainability_hooks = explainability_hooks or []

        if self.optimizer is None:
            from omegaconf import DictConfig

            args = optimizer_args
            if isinstance(args, DictConfig):
                pass
            final_args = args or {"lr": 1e-3}
            self.optimizer = optimizer_cls(self.model.parameters(), **final_args)

    def _handle_training_step(self, batch: Any, step: int, epoch: int) -> None:
        """Handle a single training step."""
        result = handle_training_step(self, batch, step, epoch)
        # Update visualization hooks after each batch
        # Gather data for hooks
        train_loss = None
        val_loss = None
        train_acc = None
        val_acc = None
        y_true = None
        y_pred = None
        images = None

        if self._current_loss_output is not None:
            train_loss = (
                self._current_loss_output.total.item()
                if hasattr(self._current_loss_output.total, "item")
                else self._current_loss_output.total
            )
        try:
            inputs, targets = self._unpack_batch(batch)
            y_true = targets.cpu().tolist() if hasattr(targets, "cpu") else targets
            images = inputs.cpu().numpy() if hasattr(inputs, "cpu") else inputs

            output = self.model(inputs.to(self.device))
            if hasattr(output, "logits"):
                logits = output.logits
            else:
                logits = output
            if logits is not None:
                y_pred = (
                    torch.argmax(logits, dim=1).cpu().tolist()
                    if hasattr(logits, "cpu")
                    else logits
                )
        except Exception:
            pass
        for viz in self.visualization_hooks:
            try:
                if isinstance(viz, LossAccuracyPlot):
                    # Only pass dummy values for now; real values should be tracked across epoch
                    viz.update(
                        train_loss or 0.0,
                        val_loss or 0.0,
                        train_acc or 0.0,
                        val_acc or 0.0,
                    )
                elif isinstance(viz, ConfusionMatrixPlot):
                    if y_true is not None and y_pred is not None:
                        viz.update(y_true, y_pred)
                elif isinstance(viz, SamplePredictionsPlot):
                    if images is not None and y_true is not None and y_pred is not None:
                        viz.update(images, y_true, y_pred)
                else:
                    from refrakt_viz.supervised.per_layer_metrics import (
                        PerLayerMetricsPlot,
                    )

                    if not isinstance(viz, PerLayerMetricsPlot):
                        viz.update()
            except Exception as e:
                logger = self._get_logger()
                if logger:
                    logger.warning(f"[VizHook] update() failed: {e}")
                else:
                    print(f"[VizHook] update() failed: {e}")
        # Optionally update explainability hooks here
        return result

    def _log_training_metrics(self, loss_output: Any, output: Any, step: int) -> None:
        """Log training metrics."""
        return log_training_metrics(self, loss_output, output, step)

    def _log_artifacts(
        self, output: Any, loss_output: Any, step: int, epoch: int
    ) -> None:
        """Log artifacts for the current step."""
        return log_artifacts(self, output, loss_output, step, epoch)

    def _handle_epoch_end(self, epoch: int, best_accuracy: float) -> float:
        """Handle end of epoch operations."""
        return handle_epoch_end(self, epoch, best_accuracy)

    def _run_explainability_hooks(self, epoch: int) -> None:
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
                input_tensor, target = sample_batch[0], (
                    sample_batch[1] if len(sample_batch) > 1 else None
                )
            elif isinstance(sample_batch, dict):
                input_tensor = sample_batch.get("image") or sample_batch.get("input")
                target = sample_batch.get("target", None)
                if input_tensor is None:
                    raise ValueError(
                        "Batch dict does not contain a valid 'image' or 'input' tensor for XAI."
                    )
            else:
                input_tensor = sample_batch
                target = None
            input_tensor = input_tensor.to(self.device)
            if target is not None:
                target = target.to(self.device)
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
                        self, "model_name", "model"
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
                        self._get_logger(),
                    )
                except Exception as e:
                    logger = self._get_logger()
                    if logger:
                        logger.warning(f"Failed to save runtime XAI info: {e}")

                input_device = input_tensor.device
                model_device = next(self.model.parameters()).device
                if input_device != model_device:
                    input_tensor = input_tensor.to(model_device)
                attributions = xai_method.explain(input_tensor, target=target)
                # Save attributions as images
                # Use registry_name from params if present, else method name, always lowercased with underscores, no 'xai' suffix
                registry_name = params.get(
                    "registry_name", params.get("method", xai_cls.__name__)
                ).replace(" ", "_")
                if registry_name.lower().endswith("xai"):
                    registry_name = registry_name[:-3]

                # Use to_snake_case for consistent naming with inference phase
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
                model_name = getattr(self.model, "model_name", None) or getattr(
                    self, "model_name", "model"
                )

                if hasattr(self, "experiment_id") and self.experiment_id:
                    dt_str = self.experiment_id
                else:
                    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

                base_dir = os.path.join(
                    "./explanations", f"{model_name}_{dt_str}", "train", registry_name
                )
                os.makedirs(base_dir, exist_ok=True)

                attr_np = attributions.detach().cpu().numpy()

                for i in range(min(attr_np.shape[0], 8)):
                    arr = attr_np[i]

                    if len(arr.shape) == 0:
                        continue
                    elif len(arr.shape) == 1 and arr.shape[0] == 1:
                        continue
                    elif len(arr.shape) == 1:
                        arr = arr.reshape(
                            (int(np.sqrt(arr.shape[0])), int(np.sqrt(arr.shape[0])))
                        )
                    elif len(arr.shape) == 2:
                        pass
                    elif len(arr.shape) == 3:
                        if arr.shape[0] == 1:
                            arr = arr[0]
                        elif arr.shape[0] == 3:
                            arr = np.transpose(arr, (1, 2, 0))
                        else:
                            continue
                    else:
                        continue

                    arr = arr - arr.min()
                    arr = arr / (arr.max() + 1e-8)
                    arr = (arr * 255).astype(np.uint8)

                    if len(arr.shape) == 2:
                        img = Image.fromarray(arr, mode="L")
                    elif len(arr.shape) == 3:
                        img = Image.fromarray(arr, mode="RGB")
                    else:
                        continue

                    img_path = os.path.join(base_dir, f"sample_{i}.png")
                    img.save(img_path)

                    npy_path = os.path.join(base_dir, f"sample_{i}.npy")
                    np.save(npy_path, attr_np[i])
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
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
        """
        import time

        start_time = time.time()
        best_accuracy = 0.0
        final_loss = 0.0
        logger = self._get_logger()

        # --- Computation graph visualization at start ---
        for viz in self.visualization_hooks:
            try:
                from refrakt_viz.supervised.computation_graph import (
                    ComputationGraphPlot,
                )

                if isinstance(viz, ComputationGraphPlot):
                    # Use a sample batch to get input_tensor
                    sample_batch = next(iter(self.train_loader))
                    if isinstance(sample_batch, (tuple, list)):
                        input_tensor = sample_batch[0].to(self.device)
                    elif isinstance(sample_batch, dict):
                        input_tensor = sample_batch["input"].to(self.device)
                    else:
                        input_tensor = sample_batch.to(self.device)
                    # Try to get model name from self.model or fallback
                    model_name = getattr(self.model, "model_name", None)
                    if model_name is None and hasattr(self, "model_name"):
                        model_name = getattr(self, "model_name", "model")
                    if model_name is None:
                        model_name = "model"
                    viz.update(self.model, input_tensor, model_name=model_name)
            except Exception as e:
                if logger:
                    logger.warning(f"[VizHook] computation_graph update failed: {e}")
                else:
                    print(f"[VizHook] computation_graph update failed: {e}")

        if logger and self.global_step == 0:
            logger.log_parameters(self.model, step=self.global_step, prefix="init_")

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(loop):
                self._current_batch = batch  # Store for artifact logging
                self._handle_training_step(batch, step, epoch)
                if self._current_loss_output is not None:
                    pass
                # --- Per-layer metrics visualization after each batch ---
                for viz in self.visualization_hooks:
                    try:
                        from refrakt_viz.supervised.per_layer_metrics import (
                            PerLayerMetricsPlot,
                        )

                        if isinstance(viz, PerLayerMetricsPlot):
                            if hasattr(self.model, "get_layer_metrics"):
                                layer_metrics = self.model.get_layer_metrics()
                                viz.update(layer_metrics)
                    except Exception as e:
                        if logger:
                            logger.warning(
                                f"[VizHook] per_layer_metrics update failed: {e}"
                            )
                        else:
                            print(f"[VizHook] per_layer_metrics update failed: {e}")

            # At end of epoch, show/save visualizations
            for viz in self.visualization_hooks:
                try:
                    from refrakt_viz.supervised.per_layer_metrics import (
                        PerLayerMetricsPlot,
                    )

                    if isinstance(viz, PerLayerMetricsPlot):
                        model_name = getattr(
                            self.model,
                            "model_name",
                            getattr(self, "model_name", "model"),
                        )
                        viz.save_with_name(model_name, mode="train")
                except Exception as e:
                    if logger:
                        logger.warning(
                            f"[VizHook] per_layer_metrics save failed (train): {e}"
                        )
                    else:
                        print(f"[VizHook] per_layer_metrics save failed (train): {e}")
            # --- Run XAI hooks at end of epoch ---
            self._run_explainability_hooks(epoch)
            best_accuracy = self._handle_epoch_end(epoch, best_accuracy)

            # Update final_loss with the current loss if available
            if self._current_loss_output is not None:
                final_loss = (
                    self._current_loss_output.total.item()
                    if hasattr(self._current_loss_output.total, "item")
                    else float(self._current_loss_output.total)
                )

        if logger:
            logger.log_parameters(self.model, step=self.global_step, prefix="final_")

        training_time = time.time() - start_time
        return {
            "best_accuracy": best_accuracy,
            "final_loss": final_loss,
            "training_time": training_time,
            "epochs_completed": num_epochs,
        }

    def evaluate(self) -> float:
        """
        Evaluate the model on the validation set.

        Returns:
            float: Validation accuracy (0.0 if no samples).
        """
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0.0
        num_batches = 0
        import torch

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)

            for batch in loop:
                inputs, targets = self._unpack_batch(batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                output = self.model(inputs)
                # Extract logits for predictions (don't call loss function!)
                if isinstance(output, ModelOutput) and output.logits is not None:
                    logits = output.logits
                elif isinstance(output, torch.Tensor):
                    logits = output
                else:
                    raise ValueError(
                        "Output does not have logits for argmax in evaluate()."
                    )

                if logits is not None:
                    preds = torch.argmax(logits, dim=1)
                else:
                    raise ValueError("Logits are None in evaluate().")

                correct += (preds == targets).sum().item()
                total += targets.size(0)
                num_batches += 1

                # Compute loss for this batch
                # Create ModelOutput for loss function if it's not already one
                if not isinstance(output, ModelOutput):
                    output_for_loss = ModelOutput(logits=logits)
                else:
                    output_for_loss = output
                loss = self.loss_fn(output_for_loss, targets)
                if hasattr(loss, "total"):
                    batch_loss = (
                        loss.total.item()
                        if hasattr(loss.total, "item")
                        else float(loss.total)
                    )
                else:
                    batch_loss = loss.item() if hasattr(loss, "item") else float(loss)
                total_loss += batch_loss

                loop.set_postfix(
                    {
                        "acc": f"{(correct / total * 100):.2f}%",
                        "loss": f"{batch_loss:.4f}",
                    }
                )

                # --- Per-layer metrics visualization after each batch (test split) ---
                from refrakt_viz.supervised.per_layer_metrics import PerLayerMetricsPlot

                for viz in self.visualization_hooks:
                    try:
                        if isinstance(viz, PerLayerMetricsPlot):
                            if hasattr(self.model, "get_layer_metrics"):
                                layer_metrics = self.model.get_layer_metrics()
                                viz.update(layer_metrics, split="test")
                    except Exception as e:
                        if logger:
                            logger.warning(
                                f"[VizHook] per_layer_metrics update failed (test): {e}"
                            )
                        else:
                            print(
                                f"[VizHook] per_layer_metrics update failed (test): {e}"
                            )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        acc = correct / total if total > 0 else 0.0

        # Get logger from artifact dumper or extra params
        logger = self._get_logger()
        if logger:
            logger.info(
                f"Validation Accuracy: {acc * 100:.2f}% | Avg Loss: {avg_loss:.4f}"
            )
        else:
            print(f"\nValidation Accuracy: {acc * 100:.2f}% | Avg Loss: {avg_loss:.4f}")

        if self.artifact_dumper:
            self.artifact_dumper.log_scalar_dict(
                {"accuracy": acc, "val_loss": avg_loss},
                step=self.global_step,
                prefix="val",
            )

        # Optionally update/show/save visualizations after evaluation
        for viz in self.visualization_hooks:
            try:
                from refrakt_viz.supervised.confusion_matrix import ConfusionMatrixPlot
                from refrakt_viz.supervised.loss_accuracy import LossAccuracyPlot
                from refrakt_viz.supervised.per_layer_metrics import PerLayerMetricsPlot

                if isinstance(viz, PerLayerMetricsPlot):
                    model_name = getattr(
                        self.model, "model_name", getattr(self, "model_name", "model")
                    )
                    viz.save_with_name(model_name, mode="test")
                elif isinstance(viz, LossAccuracyPlot):
                    model_name = getattr(
                        self.model, "model_name", getattr(self, "model_name", "model")
                    )
                    # Update with zeros for train, real values for val
                    viz.update(0.0, avg_loss, 0.0, acc)
                    viz.save_with_name(model_name, mode="test")
                elif isinstance(viz, ConfusionMatrixPlot):
                    model_name = getattr(
                        self.model, "model_name", getattr(self, "model_name", "model")
                    )
                    viz.save_with_name(model_name, mode="test")
                # Do not save SamplePredictionsPlot here
            except Exception as e:
                if logger:
                    logger.warning(f"[VizHook] visualization save failed (test): {e}")
                else:
                    print(f"[VizHook] visualization save failed (test): {e}")

        return acc

    def _unpack_batch(
        self, batch: Union[Tuple[Any, Any], List[Any], Dict[str, torch.Tensor]]
    ) -> Tuple[Any, Any]:
        """
        Unpack a batch into input and target tensors.

        Args:
            batch (Union[tuple, list, Dict[str, torch.Tensor]]): Batch from DataLoader.

        Returns:
            tuple: (inputs, targets)

        Raises:
            TypeError: If the batch format is unsupported.
        """
        if isinstance(batch, (tuple, list)):
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch["input"], batch["target"]
        raise TypeError("Unsupported batch format")

    def _get_logger(self) -> Optional[Any]:
        """
        Retrieve the logger from the artifact dumper or extra parameters.

        Returns:
            Any: Logger object if available, else None.
        """
        if self.artifact_dumper and hasattr(self.artifact_dumper, "logger"):
            return self.artifact_dumper.logger
        return self.extra_params.get("logger")
