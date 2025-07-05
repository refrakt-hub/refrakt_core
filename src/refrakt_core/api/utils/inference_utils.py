"""
Inference utilities.

This module contains utility functions for inference operations.
"""

import os
import torch
from typing import Any, Dict, Optional, Union, cast

from omegaconf import DictConfig, OmegaConf
from refrakt_core.api.core.logger import RefraktLogger


def resolve_model_name_for_inference(config: DictConfig) -> str:
    """Resolve model name for inference."""
    if config.model.name == "autoencoder":
        variant = config.model.params.get("variant", "simple")
        return f"autoencoder_{variant}"
    return config.model.name


def handle_pure_ml_inference(config: DictConfig, resolved_model_name: str, logger: RefraktLogger) -> Dict[str, Any]:
    """Handle pure ML inference pipeline."""
    import joblib
    import numpy as np
    
    # Load pipeline
    save_dir = config.trainer.params.save_dir if hasattr(config.trainer, 'params') and hasattr(config.trainer.params, 'save_dir') else './checkpoints'
    pipeline_path = os.path.join(save_dir, f"{resolved_model_name}_ml.joblib")
    pipeline = joblib.load(pipeline_path)
    feature_pipeline = pipeline['feature_pipeline']
    ml_model = pipeline['model']
    
    # Load data
    from refrakt_core.api.utils.train_utils import build_ml_numpy_splits
    _, _, X_val, y_val = build_ml_numpy_splits(config)
    preds = ml_model.predict(feature_pipeline.transform(X_val))
    acc = (preds == y_val).mean() if y_val is not None else None
    logger.info(f"[ML] Inference complete. Accuracy: {acc}")
    
    return {
        'model': ml_model,
        'feature_pipeline': feature_pipeline,
        'preds': preds,
        'y_true': y_val,
        'accuracy': acc,
        'config': config,
    }


def setup_model_and_device(config: DictConfig, device: torch.device) -> tuple:
    """Setup model and device for inference."""
    from refrakt_core.registry.model_registry import get_model
    from refrakt_core.registry.wrapper_registry import get_wrapper
    from refrakt_core.api.builders.model_builder import build_model

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


def load_fusion_head_if_provided(fusion_head_path: Optional[str], logger: RefraktLogger) -> Optional[Any]:
    """Load fusion head if path is provided."""
    if fusion_head_path and os.path.exists(fusion_head_path):
        from refrakt_core.api.utils.train_utils import load_fusion_head
        fusion_head = load_fusion_head(fusion_head_path)
        logger.info(f"Loaded fusion head from {fusion_head_path}")
        return fusion_head
    return None


def extract_inputs_from_batch(batch: Any) -> Optional[torch.Tensor]:
    """Extract inputs from batch."""
    if isinstance(batch, torch.Tensor):
        return batch
    elif isinstance(batch, dict):
        inputs = batch.get("input") or batch.get("image") or batch.get("lr")
        if inputs is None:
            raise ValueError("No valid input key found in batch.")
        return inputs
    return None


def run_inference_loop(model: Any, data_loader: Any) -> list:
    """Run inference loop."""
    results = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs = extract_inputs_from_batch(batch)
            if inputs is None:
                continue
            outputs = model(inputs)
            results.append(outputs)
    return results 