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
import sys
import traceback
import warnings
import os
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.helpers.inference_helpers import (
    _check_pure_ml_inference,
    _load_and_validate_config,
    _load_model_and_setup,
    _setup_data_loader,
    _setup_device,
    _setup_logging,
)
from refrakt_core.api.utils.inference_utils import (
    handle_pure_ml_inference,
    load_fusion_head_if_provided,
    resolve_model_name_for_inference,
    run_inference_loop,
)
from refrakt_core.api.utils.pipeline_utils import parse_runtime_hooks
from refrakt_core.api.utils.hooks_orchestrator import (  # type: ignore
    instantiate_visualization_hooks,
    instantiate_explainability_hooks,
)

warnings.filterwarnings("ignore")

gc.collect()
torch.cuda.empty_cache()


def _run_inference_explainability_hooks(
    xai_hooks: Any,
    model: Any,
    data_loader: Any,
    device: Any,
    explainability_samples: Union[str, int] = 'all',
) -> None:
    """
    Run XAI hooks on inference samples and save attributions as images.
    Args:
        xai_hooks: List of (xai_cls, params) tuples
        model: Trained model
        data_loader: Inference data loader
        device: Device for computation
        explainability_samples: 'all' or int (number of samples to explain)
    """
    if not xai_hooks:
        return
    # Prepare sample indices
    if explainability_samples == 'all':
        sample_indices = None  # All samples
    else:
        try:
            N = int(explainability_samples)
        except Exception:
            N = 8
        sample_indices = set(range(N))
    sample_count = 0
    for batch in data_loader:
        if isinstance(batch, (tuple, list)):
            input_tensor = batch[0]
            target = batch[1] if len(batch) > 1 else None
        elif isinstance(batch, dict):
            input_tensor = batch["input"]
            target = batch.get("target", None)
        else:
            input_tensor = batch
            target = None
        input_tensor = input_tensor.to(device)
        if target is not None:
            target = target.to(device)
        batch_size = input_tensor.shape[0]
        for xai_cls, params in xai_hooks:
            try:
                xai_instance = xai_cls(model, **params)
                attributions = xai_instance.explain(input_tensor, target=target)
                method = xai_cls.__name__
                save_dir = os.path.join("./explanations_inference", method)
                os.makedirs(save_dir, exist_ok=True)
                attr_np = attributions.detach().cpu().numpy()
                for i in range(batch_size):
                    if sample_indices is not None and sample_count not in sample_indices:
                        sample_count += 1
                        continue
                    arr = attr_np[i]
                    arr = arr - arr.min()
                    arr = arr / (arr.max() + 1e-8)
                    arr = (arr * 255).astype(np.uint8)
                    if arr.shape[0] == 1:
                        arr = arr[0]
                    elif arr.shape[0] == 3:
                        arr = np.transpose(arr, (1, 2, 0))
                    img = Image.fromarray(arr)
                    img.save(os.path.join(save_dir, f"sample_{sample_count}.png"))
                    sample_count += 1
                    if sample_indices is not None and sample_count >= max(sample_indices) + 1:
                        return
            except Exception as e:
                print(f"[XAI-Inference] Failed to run or save explainability for {xai_cls}: {e}")
        if sample_indices is not None and sample_count >= max(sample_indices) + 1:
            break


def inference(
    cfg: Union[str, DictConfig],
    model_path: str,
    fusion_head_path: Optional[str] = None,
    data: Any = None,
    logger: Optional[RefraktLogger] = None,
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
        setup_artifact_dumper(config, resolved_model_name, logger)

        # --- Inference-time hooks ---
        from omegaconf import OmegaConf
        config_dict = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config_dict, dict):
            config_dict = {}
        from typing import cast
        viz_hooks, xai_hooks = parse_runtime_hooks(cast(Dict[str, Any], config_dict))
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
                input_tensor = batch[0] if isinstance(batch, (tuple, list)) else batch
                input_tensor = input_tensor.to(device)
                output = model(input_tensor)
            # Collect sample predictions if possible
            if sample_pred_plot is not None:
                # Try to extract images, y_true, y_pred from batch/output
                try:
                    images = batch[0].cpu().numpy() if isinstance(batch, (tuple, list)) and hasattr(batch[0], 'cpu') else None
                    y_true = batch[1].cpu().tolist() if isinstance(batch, (tuple, list)) and hasattr(batch[1], 'cpu') else None
                    if hasattr(output, 'logits'):
                        logits = output.logits
                    else:
                        logits = output
                    if logits is not None:
                        y_pred = torch.argmax(logits, dim=1).cpu().tolist() if hasattr(logits, 'cpu') else None
                    else:
                        y_pred = None
                    if images is not None and y_true is not None and y_pred is not None:
                        sample_pred_plot.update(images, y_true, y_pred)
                except Exception as e:
                    print(f"[SamplePredictionsPlot] update() failed: {e}")
            results.append(output)

        # --- Run XAI after inference if enabled ---
        if xai_components:
            # For each XAI method, get its per-method no_samples value (default 'all')
            for (xai_cls, params), hook_cfg in zip(xai_components, xai_hook_dicts):
                no_samples = hook_cfg.get('no_samples', 'all')
                _run_inference_explainability_hooks([(xai_cls, params)], model, data_loader, device, no_samples)
        # Save sample predictions plot at the end of inference
        if sample_pred_plot is not None:
            model_name = getattr(model, 'model_name', resolved_model_name)
            sample_pred_plot.save_with_name(model_name)

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


# inference._load_and_validate_config = _load_and_validate_config  # type: ignore

__all__ = ["inference", "_load_and_validate_config"]
