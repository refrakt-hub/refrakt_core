"""
Inference utilities for Refrakt.

This module provides comprehensive utility functions for inference operations,
including model and device setup, batch extraction, and running inference loops.
It supports both deep learning and pure ML inference pipelines, and provides
helpers for extracting model inputs and outputs in a robust and type-safe manner.

The module handles:
- Model name resolution for inference operations
- Pure ML inference pipeline handling
- Model and device setup for inference
- Input extraction from various batch formats
- Inference loop execution and result collection
- Fusion head loading and management
- Configuration validation and processing

These utilities ensure robust inference pipeline operations with proper error handling,
automatic input processing, and comprehensive result collection capabilities.

Typical usage involves calling these utility functions to set up and execute
complete inference pipelines with automatic input processing and result collection.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.core.logger import RefraktLogger


def resolve_model_name_for_inference(config: DictConfig) -> str:
    """
    Resolve the model name for inference, handling variants and autoencoders.

    Args:
        config (DictConfig): The configuration object.

    Returns:
        str: The resolved model name.
    """
    if config.model.name == "autoencoder":
        variant = config.model.params.get("variant", "simple")
        return f"autoencoder_{variant}"
    return str(config.model.name)


def handle_pure_ml_inference(
    config: DictConfig, resolved_model_name: str, logger: RefraktLogger
) -> Dict[str, Any]:
    """
    Handle pure ML inference pipeline, loading the pipeline and running predictions.

    Args:
        config (DictConfig): The configuration object.
        resolved_model_name (str): The resolved model name.
        logger (RefraktLogger): Logger instance.

    Returns:
        Dict[str, Any]: Dictionary with model, pipeline, predictions, and accuracy.
    """
    import joblib  # type: ignore[import-untyped]

    # Load pipeline
    save_dir = (
        config.trainer.params.save_dir
        if hasattr(config.trainer, "params")
        and hasattr(config.trainer.params, "save_dir")
        else "./checkpoints"
    )
    pipeline_path = os.path.join(save_dir, f"{resolved_model_name}_ml.joblib")
    pipeline = joblib.load(pipeline_path)
    feature_pipeline = pipeline["feature_pipeline"]
    ml_model = pipeline["model"]

    # Load data
    from refrakt_core.api.utils.train_utils import build_ml_numpy_splits

    _, _, X_val, y_val = build_ml_numpy_splits(config)
    preds = ml_model.predict(feature_pipeline.transform(X_val))
    acc = (preds == y_val).mean() if y_val is not None else None
    logger.info(f"[ML] Inference complete. Accuracy: {acc}")

    return {
        "model": ml_model,
        "feature_pipeline": feature_pipeline,
        "preds": preds,
        "y_true": y_val,
        "accuracy": acc,
        "config": config,
    }


def setup_model_and_device(
    config: DictConfig, device: torch.device
) -> Tuple[Any, Dict[str, Any]]:
    """
    Setup model and device for inference.

    Args:
        config (DictConfig): The configuration object.
        device (torch.device): The device to use.

    Returns:
        Tuple[Any, Dict[str, Any]]: The model and modules dictionary.
    """
    from refrakt_core.api.builders.model_builder import build_model
    from refrakt_core.registry.model_registry import get_model
    from refrakt_core.registry.wrapper_registry import get_wrapper

    modules = {
        "get_model": get_model,
        "get_wrapper": get_wrapper,
    }

    model_cls = get_model(config.model.name)
    model = build_model(
        cast(OmegaConf, config),
        modules={
            "get_model": get_model,
            "get_wrapper": get_wrapper,
            "model": model_cls,
        },
        device=str(device),
    )

    return model, modules


def load_fusion_head_if_provided(
    fusion_head_path: Optional[str], logger: RefraktLogger
) -> Optional[Any]:
    """
    Load fusion head if a path is provided and exists.

    Args:
        fusion_head_path (Optional[str]): Path to the fusion head file.
        logger (RefraktLogger): Logger instance.

    Returns:
        Optional[Any]: The loaded fusion head, or None if not found.
    """
    if fusion_head_path and os.path.exists(fusion_head_path):
        from refrakt_core.api.utils.train_utils import load_fusion_head

        fusion_head = load_fusion_head(fusion_head_path)
        logger.info(f"Loaded fusion head from {fusion_head_path}")
        return fusion_head
    return None


def extract_inputs_from_batch(batch: Any) -> Optional[torch.Tensor]:
    """
    Extract inputs from a batch for inference.

    Args:
        batch (Any): The batch, which may be a Tensor, dict, or other type.

    Returns:
        Optional[torch.Tensor]: The input tensor, or None if not found.
    """
    if isinstance(batch, torch.Tensor):
        return batch
    elif isinstance(batch, dict):
        for key in ["input", "image", "lr"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return cast(torch.Tensor, batch[key])
        return None
    return None


def run_inference_loop(model: torch.nn.Module, data_loader: Any) -> List[torch.Tensor]:
    """
    Run inference loop over a data loader.

    Args:
        model (torch.nn.Module): The model to use for inference.
        data_loader (Any): The data loader to iterate over.

    Returns:
        List[torch.Tensor]: List of model outputs for each batch.
    """
    results = []
    with torch.no_grad():
        for _i, batch in enumerate(data_loader):
            inputs = extract_inputs_from_batch(batch)
            if inputs is None:
                continue
            outputs = model(inputs)
            results.append(cast(torch.Tensor, outputs))
    return results
