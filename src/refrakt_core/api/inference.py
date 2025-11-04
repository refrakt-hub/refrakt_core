"""
Inference API for Refrakt.

This module provides the main inference function for running predictions using trained
models in the Refrakt framework. It orchestrates the complete inference pipeline
including model loading, data preparation, prediction execution, and results handling.

The module handles:
- Configuration loading and validation
- Model checkpoint loading and setup
- Data preprocessing and dataloader creation
- Inference execution with batch processing
- Fusion model integration (if applicable)
- Results formatting and return
- Error handling and cleanup
- Memory management for GPU inference
"""

import gc
import json
import os
import re
import sys
import traceback
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.helpers.inference_helpers import (
    _check_pure_ml_inference,
    _load_and_validate_config,
    _load_model_and_setup,
    _setup_data_loader,
    _setup_device,
    _setup_logging,
)
from refrakt_core.api.utils.hooks_orchestrator import (  # type: ignore
    instantiate_explainability_hooks,
    instantiate_visualization_hooks,
)
from refrakt_core.api.utils.inference_utils import (
    handle_pure_ml_inference,
    load_fusion_head_if_provided,
    resolve_model_name_for_inference,
    run_inference_loop,
)
from refrakt_core.api.utils.pipeline_utils import parse_runtime_hooks
from refrakt_core.error_handling import XAINotSupportedError

warnings.filterwarnings("ignore")

gc.collect()
torch.cuda.empty_cache()


def to_snake_case(name):
    """Convert camelCase or PascalCase to snake_case."""
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


def _run_inference_explainability_hooks(
    xai_hooks: Any,
    model: Any,
    data_loader: Any,
    device: Any,
    explainability_samples: Union[str, int] = "all",
    resolved_model_name: Optional[str] = None,
    base_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
    logger: Optional[RefraktLogger] = None,
) -> None:
    """
    Run XAI hooks on inference samples and save attributions as images.
    Args:
        xai_hooks: List of (xai_cls, params) tuples
        model: Trained model
        data_loader: Inference data loader
        device: Device for computation
        explainability_samples: 'all' or int (number of samples to explain)
        resolved_model_name: Name of the model
        base_dir: Pre-created base directory (if None, will create one)
        experiment_id: Unique experiment ID to ensure consistent directory naming
    """
    if not xai_hooks:
        return

    # Check if this is a contrastive learning model once at the beginning
    model_name = getattr(model, "model_name", "unknown")
    model_type = getattr(model, "model_type", "unknown")
    wrapper_name = getattr(model, "wrapper_name", "unknown")

    # Check for contrastive indicators in model name/type/wrapper
    contrastive_indicators = ["simclr", "dino", "msn", "contrastive"]

    is_contrastive = any(
        indicator in model_name.lower()
        or indicator in model_type.lower()
        or indicator in wrapper_name.lower()
        for indicator in contrastive_indicators
    )

    # If it's a contrastive model, print warning once and skip all XAI hooks
    if is_contrastive:
        if logger:
            logger.warning(
                "⚠️  XAI components are currently not supported for contrastive family models "
                "(SimCLR, DINO, MSN) in refrakt v1. Skipping all XAI hooks during inference."
            )
        return

    # Prepare sample indices
    if explainability_samples == "all":
        sample_indices = None  # All samples
    else:
        try:
            N = int(explainability_samples)
        except Exception:
            N = 8
        sample_indices = set(range(N))

    # Create base directory if not provided
    if base_dir is None:
        # Use experiment_id if provided, otherwise create timestamp
        if experiment_id:
            dir_name = f"{resolved_model_name or getattr(model, 'model_name', 'model')}_{experiment_id}"
        else:
            dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"{resolved_model_name or getattr(model, 'model_name', 'model')}_{dt_str}"

        # Use explanations directory structure for inference
        base_dir = os.path.join("./explanations", dir_name, "inference")
        os.makedirs(base_dir, exist_ok=True)

    sample_count = 0

    # Process each XAI method
    for xai_cls, params in xai_hooks:
        method = "Unknown"  # Default method name
        try:
            # Create XAI instance
            if xai_cls.__name__ == "ConceptSaliencyXAI":
                xai_instance = xai_cls(
                    model, dataloader=data_loader, device=device, **params
                )
            else:
                xai_instance = xai_cls(model, **params)

            # Save runtime XAI info for metadata collection
            try:
                from refrakt_cli.helpers.shared_core import save_runtime_xai_info

                # Use resolved model name for base directory
                model_name = resolved_model_name or getattr(
                    model, "model_name", "model"
                )
                if experiment_id:
                    checkpoints_base_dir = f"./checkpoints/{model_name}_{experiment_id}"
                    save_runtime_xai_info(
                        xai_instance,
                        xai_cls.__name__,
                        params,
                        checkpoints_base_dir,
                        logger,
                    )
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save runtime XAI info: {e}")

            # Determine method name and registry name
            method = xai_cls.__name__
            registry_name = params.get(
                "registry_name", params.get("method", xai_cls.__name__)
            ).replace(" ", "_")
            if registry_name.lower().endswith("xai"):
                registry_name = registry_name[:-3]
            registry_name = to_snake_case(registry_name)

            # Create method-specific directory
            save_dir = os.path.join(base_dir, registry_name)
            os.makedirs(save_dir, exist_ok=True)

            if logger:
                logger.debug(
                    f"[XAI-Inference] Processing {method} (registry: {registry_name})"
                )

            # Process batches
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    input_tensor = batch[0]
                    target = batch[1] if len(batch) > 1 else None
                elif isinstance(batch, dict):
                    input_tensor = batch.get("input") or batch.get("lr")
                    if input_tensor is None:
                        if logger:
                            logger.error(
                                f"[XAI-Inference] Batch dict does not contain 'input' or 'lr' key. Available keys: {list(batch.keys())}"
                            )
                        continue
                    target = batch.get("target") or batch.get("hr")
                else:
                    input_tensor = batch
                    target = None

                input_tensor = input_tensor.to(device)
                if target is not None:
                    target = target.to(device)

                batch_size = input_tensor.shape[0]

                # Get attributions for this batch
                attributions = xai_instance.explain(input_tensor, target=target)
                attr_np = attributions.detach().cpu().numpy()

                if logger:
                    logger.debug(
                        f"[XAI-Inference] Batch attributions shape: {attr_np.shape}"
                    )

                # Process each sample in the batch
                for i in range(batch_size):
                    if (
                        sample_indices is not None
                        and sample_count not in sample_indices
                    ):
                        sample_count += 1
                        continue

                    arr = attr_np[i]
                    if logger:
                        logger.debug(f"[XAI-Inference] Sample {i} shape: {arr.shape}")

                    # Handle different attribution shapes
                    if len(arr.shape) == 0:  # Scalar
                        if logger:
                            logger.debug(f"[XAI-Inference] Skipping scalar attribution")
                        continue
                    elif (
                        len(arr.shape) == 1 and arr.shape[0] == 1
                    ):  # [1] - class-specific attribution
                        if logger:
                            logger.debug(
                                f"[XAI-Inference] Skipping class-specific attribution"
                            )
                        continue
                    elif len(arr.shape) == 1:  # 1D array
                        # Reshape to 2D for visualization
                        arr = arr.reshape(
                            (int(np.sqrt(arr.shape[0])), int(np.sqrt(arr.shape[0])))
                        )
                        if logger:
                            logger.debug(
                                f"[XAI-Inference] Reshaped 1D to 2D: {arr.shape}"
                            )
                    elif len(arr.shape) == 2:  # 2D array (spatial heatmap)
                        if logger:
                            logger.debug(
                                f"[XAI-Inference] Processing 2D heatmap: {arr.shape}"
                            )
                        pass  # Keep as is
                    elif len(arr.shape) == 3:  # 3D array
                        if arr.shape[0] == 1:  # [1, H, W]
                            arr = arr[0]
                            if logger:
                                logger.debug(
                                    f"[XAI-Inference] Extracted single channel: {arr.shape}"
                                )
                        elif arr.shape[0] == 3:  # [3, H, W] - RGB
                            arr = np.transpose(arr, (1, 2, 0))
                            if logger:
                                logger.debug(
                                    f"[XAI-Inference] Transposed RGB: {arr.shape}"
                                )
                        else:
                            if logger:
                                logger.debug(
                                    f"[XAI-Inference] Skipping unexpected 3D shape: {arr.shape}"
                                )
                            continue
                    else:
                        if logger:
                            logger.debug(
                                f"[XAI-Inference] Skipping unexpected shape: {arr.shape}"
                            )
                        continue

                    # Normalize and convert to image
                    arr = arr - arr.min()
                    arr = arr / (arr.max() + 1e-8)

                    # Convert to uint8 for saving
                    arr_uint8 = (arr * 255).astype(np.uint8)

                    # Save as image
                    filename = f"{registry_name}_inference_{sample_count}.png"
                    filepath = os.path.join(save_dir, filename)

                    # Use PIL to save the image
                    from PIL import Image

                    img = Image.fromarray(arr_uint8)
                    img.save(filepath)

                    if logger:
                        logger.debug(
                            f"[XAI-Inference] Saved sample {sample_count} to {filepath}"
                        )

                    sample_count += 1

                    # Check if we've reached the sample limit
                    if (
                        sample_indices is not None
                        and sample_count >= max(sample_indices) + 1
                    ):
                        if logger:
                            logger.debug(
                                f"[XAI-Inference] Reached sample limit, stopping"
                            )
                        return

                    # For explanations_inference, only save a single random sample (regardless of shape)
                    if sample_indices is not None:
                        break  # Only process one sample per batch for explanations_inference

                # For explanations_inference, only process one batch
                if sample_indices is not None:
                    break

        except Exception as e:
            if logger:
                logger.error(f"[XAI-Inference] Error processing {method}: {e}")
            import traceback

            traceback.print_exc()
            continue


def inference(
    cfg: Union[str, DictConfig],
    model_path: str,
    fusion_head_path: Optional[str] = None,
    data: Any = None,
    logger: Optional[RefraktLogger] = None,
    experiment_id: Optional[str] = None,
    config_path: Optional[str] = None,  # <-- Fix type
) -> Dict[str, Any]:
    """
    Orchestrate the inference pipeline for Refrakt.

    This function serves as the main entry point for running inference using trained
    models in the Refrakt framework. It handles the complete inference workflow
    including model loading, data preparation, prediction execution, and results
    formatting.

    The function supports both deep learning and pure ML inference modes, automatically
    detecting the appropriate pipeline based on configuration. It also supports
    fusion model inference when applicable.

    Args:
        cfg: Path to configuration file (str) or DictConfig object containing all
            inference parameters including model, data preprocessing, and inference
            settings
        model_path: Path to the trained model checkpoint file
        fusion_head_path: Optional path to fusion head checkpoint for ensemble models.
            If provided, the fusion head will be loaded and used for predictions
        data: Optional custom data for inference. Can be a dataset, dataloader, or
            raw data. If None, data will be loaded from configuration
        logger: Optional RefraktLogger instance for logging. If None, a new logger
            will be created based on configuration
        experiment_id: Optional experiment ID for consistent directory naming across
            pipeline components
        config_path: Optional path to the config file for metadata/summary_metrics

    Returns:
        Dictionary containing inference results with the following keys:
        - 'model': The loaded model instance
        - 'results': Prediction results and outputs
        - 'config': The configuration used for inference

    Raises:
        SystemExit: If inference fails due to configuration errors, model loading
            issues, or other critical failures. The function will log detailed error
            information before exiting.
    """
    try:
        # Load and validate configuration
        config = _load_and_validate_config(cfg)
        resolved_model_name = resolve_model_name_for_inference(config)

        # Setup logging
        logger = _setup_logging(config, resolved_model_name, logger)

        # Check for pure ML inference
        if _check_pure_ml_inference(config):
            return handle_pure_ml_inference(config, resolved_model_name, logger)

        # Setup device
        device = _setup_device()

        # Load model and setup
        model, modules = _load_model_and_setup(config, device, model_path, logger)

        # Load fusion head if provided
        load_fusion_head_if_provided(fusion_head_path, logger)

        # Setup data loader
        data_loader = _setup_data_loader(config, data, logger)

        # Setup artifact dumper
        from refrakt_core.api.utils.train_utils import setup_artifact_dumper

        setup_artifact_dumper(config, resolved_model_name, logger, experiment_id)

        # --- Inference-time hooks ---
        from omegaconf import OmegaConf

        config_dict = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config_dict, dict):
            config_dict = {}
        from typing import cast

        viz_hooks, xai_hooks, explain_flag = parse_runtime_hooks(
            cast(Dict[str, Any], config_dict)
        )
        # Convert xai_hooks to list of dicts if needed
        xai_hook_dicts = []
        for hook in xai_hooks:
            if isinstance(hook, dict):
                xai_hook_dicts.append(hook)
            elif isinstance(hook, str):
                xai_hook_dicts.append({"method": hook})
        xai_components = instantiate_explainability_hooks(xai_hook_dicts)
        # --- End hooks ---

        # --- Inference-time sample predictions visualization ---
        from refrakt_viz.supervised.sample_predictions import SamplePredictionsPlot

        class_names = None
        if hasattr(config, "dataset") and hasattr(config.dataset, "params"):
            class_names = getattr(config.dataset.params, "class_names", None)
        sample_pred_plot = None
        if class_names:
            sample_pred_plot = SamplePredictionsPlot(class_names=class_names)

        # Run inference (with sample predictions visualization)
        results = []
        for batch in data_loader:
            # Standard inference
            with torch.no_grad():
                # Handle different batch formats
                if isinstance(batch, dict):
                    # For SuperResolution datasets or similar dict-based formats
                    if "lr" in batch:
                        input_tensor = batch["lr"]
                    elif "input" in batch:
                        input_tensor = batch["input"]
                    else:
                        # Use the first available tensor value
                        input_tensor = next(iter(batch.values()))
                elif isinstance(batch, (tuple, list)):
                    input_tensor = batch[0]
                else:
                    input_tensor = batch

                input_tensor = input_tensor.to(device)
                output = model(input_tensor)
            # Collect sample predictions if possible
            if sample_pred_plot is not None:
                # Try to extract images, y_true, y_pred from batch/output
                try:
                    # Handle different batch formats for extracting images and labels
                    if isinstance(batch, dict):
                        # For dict-based datasets, try to extract images and labels
                        if "lr" in batch:
                            images = (
                                batch["lr"].cpu().numpy()
                                if hasattr(batch["lr"], "cpu")
                                else None
                            )
                        elif "input" in batch:
                            images = (
                                batch["input"].cpu().numpy()
                                if hasattr(batch["input"], "cpu")
                                else None
                            )
                        else:
                            images = (
                                next(iter(batch.values())).cpu().numpy()
                                if hasattr(next(iter(batch.values())), "cpu")
                                else None
                            )

                        # Try to find ground truth labels
                        y_true = None
                        if "hr" in batch:
                            y_true = (
                                batch["hr"].cpu().tolist()
                                if hasattr(batch["hr"], "cpu")
                                else None
                            )
                        elif "target" in batch or "label" in batch:
                            target_key = "target" if "target" in batch else "label"
                            y_true = (
                                batch[target_key].cpu().tolist()
                                if hasattr(batch[target_key], "cpu")
                                else None
                            )
                    elif isinstance(batch, (tuple, list)):
                        images = (
                            batch[0].cpu().numpy() if hasattr(batch[0], "cpu") else None
                        )
                        y_true = (
                            batch[1].cpu().tolist()
                            if isinstance(batch, (tuple, list))
                            and len(batch) > 1
                            and hasattr(batch[1], "cpu")
                            else None
                        )
                    else:
                        images = batch.cpu().numpy() if hasattr(batch, "cpu") else None
                        y_true = None

                    if hasattr(output, "logits"):
                        logits = output.logits
                    else:
                        logits = output
                    if logits is not None:
                        y_pred = (
                            torch.argmax(logits, dim=1).cpu().tolist()
                            if hasattr(logits, "cpu")
                            else None
                        )
                    else:
                        y_pred = None
                    if images is not None and y_true is not None and y_pred is not None:
                        sample_pred_plot.update(images, y_true, y_pred)
                except Exception as e:
                    if logger:
                        logger.error(f"[SamplePredictionsPlot] update() failed: {e}")
                    else:
                        print(f"[SamplePredictionsPlot] update() failed: {e}")
            results.append(output)

        # --- Run XAI after inference if enabled ---
        if xai_components:
            # Use the experiment_id passed from the pipeline
            if experiment_id is None:
                experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                if logger:
                    logger.warning(
                        f"No experiment_id provided for inference, generated new one: {experiment_id}"
                    )
            # Use explanations directory structure for inference
            shared_base_dir = os.path.join(
                "./explanations",
                f"{resolved_model_name or getattr(model, 'model_name', 'model')}_{experiment_id}",
                "inference",
            )
            os.makedirs(shared_base_dir, exist_ok=True)

            # For each XAI method, get its per-method no_samples value (default 1)
            for (xai_cls, params), hook_cfg in zip(xai_components, xai_hook_dicts):
                no_samples = hook_cfg.get("no_samples", 1)  # Default to 1 if not set
                _run_inference_explainability_hooks(
                    [(xai_cls, params)],
                    model,
                    data_loader,
                    device,
                    no_samples,
                    resolved_model_name,
                    shared_base_dir,
                    experiment_id,
                    logger,
                )
        # Save sample predictions plot at the end of inference
        if sample_pred_plot is not None:
            model_name = getattr(model, "model_name", resolved_model_name)
            sample_pred_plot.save_with_name(model_name)

        # At the end, save summary_metrics.json with config_path if needed
        from refrakt_core.api.helpers.train_helpers import (
            _save_inference_summary_metrics,
        )

        _save_inference_summary_metrics(
            model,
            results,
            resolved_model_name,
            logger,
            experiment_id,
            [config_path] if config_path else [],
        )

        logger.info("\n✅ Inference completed successfully!")
        return {
            "model": model,
            "results": results,
            "config": config,
        }

    except Exception as e:
        if not (logger and hasattr(logger, "error")):
            logger = None
        if logger:
            logger.error(f"\n❌ Inference failed: {str(e)}")
            logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        gc.collect()
        torch.cuda.empty_cache()


__all__ = ["inference", "_load_and_validate_config"]
