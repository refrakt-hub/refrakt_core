"""
Utility functions for testing in Refrakt.

Includes config loading, test dataloader building, and model checkpoint loading helpers.
"""

import glob
import os
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.train_utils import analyze_and_resize_dataset_images
from refrakt_core.utils.methods import extract_visual_tensor

"""
Test utilities for Refrakt API.

This module provides utility functions for testing operations including
model name resolution, fusion evaluation, batch data extraction, and manual evaluation.
"""

import os
import torch
from typing import Any, Dict, Optional, Tuple, cast
from omegaconf import DictConfig

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.integrations.fusion.trainer import FusionTrainer
from refrakt_core.integrations.fusion.builder import build_fusion_head
from refrakt_core.integrations.cpu.wrapper import SklearnWrapper
from refrakt_core.integrations.gpu.wrapper import CuMLWrapper


def _load_config(cfg: Any) -> Any:
    """
    Load an OmegaConf config from a file path or return as-is if already a config.

    Args:
        cfg (Any): Path to config file or OmegaConf config.

    Returns:
        Any: Loaded configuration object (DictConfig or ListConfig).
    """
    return OmegaConf.load(cfg) if isinstance(cfg, str) else cfg


def _build_test_loader(config: Any) -> Any:
    """
    Build a test dataloader from the given config.

    Args:
        config (Any): Configuration object (DictConfig or ListConfig).

    Returns:
        Any: PyTorch DataLoader for test data.
    """
    test_cfg = OmegaConf.merge(
        config.dataset, OmegaConf.create({"params": {"train": False}})
    )
    # Ensure test_cfg is a DictConfig
    from omegaconf import ListConfig

    if isinstance(test_cfg, ListConfig):
        test_cfg = OmegaConf.create(OmegaConf.to_container(test_cfg, resolve=True))
    if not isinstance(test_cfg, DictConfig):
        raise TypeError("test_cfg must be a DictConfig after conversion.")
    dataset = build_dataset(test_cfg)
    return build_dataloader(dataset, config.dataloader)


def _build_test_loader_with_resize(config: Any, logger: RefraktLogger) -> Any:
    """
    Build a test dataloader from the given config with automatic image resizing.

    Args:
        config (Any): Configuration object (DictConfig or ListConfig).
        logger (RefraktLogger): Logger instance for logging resize operations.

    Returns:
        Any: PyTorch DataLoader for test data with resizing applied if needed.
    """
    test_cfg = OmegaConf.merge(
        config.dataset, OmegaConf.create({"params": {"train": False}})
    )
    # Ensure test_cfg is a DictConfig
    from omegaconf import ListConfig

    if isinstance(test_cfg, ListConfig):
        test_cfg = OmegaConf.create(OmegaConf.to_container(test_cfg, resolve=True))
    if not isinstance(test_cfg, DictConfig):
        raise TypeError("test_cfg must be a DictConfig after conversion.")
    
    # Build dataset
    dataset = build_dataset(test_cfg)
    
    # Analyze and resize if needed
    test_resized, dataset = analyze_and_resize_dataset_images(dataset, logger)
    
    if test_resized:
        logger.info("🔄 Using resized test dataset")
    
    return build_dataloader(dataset, config.dataloader)


def _load_model_checkpoint(
    model: torch.nn.Module,
    model_path: Optional[str],
    device: torch.device,
    logger: Any,
) -> int:
    """
    Load a model checkpoint, with fallback logic for missing files and variants.

    Args:
        model (torch.nn.Module): The model to load weights into.
        model_path (Optional[str]): Path to the checkpoint file.
        device (torch.device): Device to map the checkpoint to.
        logger (Any): Logger for warnings and info.

    Returns:
        int: The global step from the checkpoint, or 0 if not found.

    Raises:
        FileNotFoundError: If no valid checkpoint is found.
    """
    import typing
    from collections import defaultdict

    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode
    from torch.serialization import add_safe_globals

    # Allow OmegaConf configs to be unpickled safely
    add_safe_globals(
        [
            ListConfig,
            DictConfig,
            ContainerMetadata,
            typing.Any,
            list,
            dict,
            defaultdict,
            int,
            float,
            AnyNode,
            Metadata,
        ]
    )

    if model_path is None:
        logger.warning("No model checkpoint provided — using random init weights")
        return 0

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.info(f"Loaded model from {model_path}")
        return checkpoint.get("global_step", 0)

    # If file doesn't exist, try fallback logic
    base_dir = os.path.dirname(model_path)
    base_name = os.path.splitext(os.path.basename(model_path))[0]  # autoencoder_simple

    # Try exact match first
    exact_match = os.path.join(base_dir, f"{base_name}.pth")
    if os.path.exists(exact_match):
        checkpoint = torch.load(exact_match, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.warning(f"⚠️ Falling back to exact checkpoint: {exact_match}")
        return checkpoint.get("global_step", 0)

    # Try matching variants like _latest, _final
    pattern = os.path.join(base_dir, f"{base_name}_*.pth")
    candidates = glob.glob(pattern)
    if candidates:
        # Prefer latest or final first
        preferred = [c for c in candidates if "latest" in c or "final" in c]
        fallback_path = max(preferred or candidates, key=os.path.getmtime)

        checkpoint = torch.load(fallback_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.warning(f"⚠️ Falling back to available checkpoint: {fallback_path}")
        return checkpoint.get("global_step", 0)

    logger.error(f"Model path does not exist: {model_path}")
    raise FileNotFoundError(model_path)


def _resolve_model_name(config: DictConfig) -> str:
    """Resolve the model name based on configuration."""
    if config.model.name == "autoencoder":
        variant = config.model.params.get("variant", "simple")
        resolved_model_name = f"autoencoder_{variant}"
    else:
        resolved_model_name = config.model.name
    
    # Check if using custom dataset and append _custom suffix
    dataset_params = config.dataset.params if hasattr(config, "dataset") and hasattr(config.dataset, "params") else {}
    dataset_path = dataset_params.get("path", "") or dataset_params.get("zip_path", "")
    if dataset_path and str(dataset_path).endswith(".zip"):
        resolved_model_name = f"{resolved_model_name}_custom"
    
    return resolved_model_name


def _handle_pure_ml_pipeline(config: DictConfig, resolved_model_name: str, logger: RefraktLogger) -> None:
    """Handle pure ML pipeline testing."""
    import joblib
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
    logger.info(f"[ML] Test complete. Accuracy: {acc}")
    print("\nEvaluation Results:", {'accuracy': acc})


def _setup_fusion_evaluation(config: DictConfig, model: torch.nn.Module, dataloader: Any, 
                           device: torch.device, artifact_dumper: Any, logger: RefraktLogger) -> Optional[float]:
    """Setup and run fusion evaluation if applicable."""
    fusion_cfg = getattr(config.model, "fusion", None)
    if not fusion_cfg:
        return None
    
    fusion_type = fusion_cfg.type
    fusion_model_key = fusion_cfg.model
    fusion_model_path = os.path.join(
        config.trainer.params.save_dir, f"{config.model.name}_fusion.joblib"
    )
    
    if not os.path.exists(fusion_model_path):
        logger.warning(f"[FUSION] No fusion model found at: {fusion_model_path}")
        return None
    
    logger.info(f"[FUSION] Found fusion head at {fusion_model_path}")
    if fusion_type == "sklearn":
        fusion_head = SklearnWrapper.load(fusion_model_key, fusion_model_path)
    elif fusion_type == "cuml":
        fusion_head = CuMLWrapper.load(fusion_model_key, fusion_model_path)
    else:
        raise ValueError(f"[FUSION] Unsupported fusion type: {fusion_type}")
    
    fusion_trainer = FusionTrainer(
        model=model,
        fusion_head=fusion_head,
        train_loader=dataloader,
        val_loader=dataloader,
        device=device,
        artifact_dumper=artifact_dumper,
        model_name=config.model.name,
    )
    fusion_acc = fusion_trainer.evaluate()
    logger.info(f"[FUSION] Validation accuracy (fusion head): {fusion_acc:.4f}")
    return fusion_acc


def _extract_batch_data(batch: Any, logger: RefraktLogger) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract inputs and targets from batch."""
    if isinstance(batch, torch.Tensor):
        return batch, None
    elif isinstance(batch, dict):
        inputs = batch.get("input") or batch.get("image") or batch.get("lr")
        targets = batch.get("target") or batch.get("label")
        if inputs is None:
            logger.warning("No valid input key found in batch, skipping...")
            return None, None
        return inputs, targets
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    else:
        logger.warning(f"Unexpected batch format: {type(batch)}, skipping...")
        return None, None


def _extract_logits(outputs: Any, logger: RefraktLogger) -> Optional[torch.Tensor]:
    """Extract logits from model outputs."""
    if hasattr(outputs, 'logits'):
        return outputs.logits
    elif isinstance(outputs, torch.Tensor):
        return outputs
    else:
        logger.warning("Could not extract logits from model output")
        return None


def _run_manual_evaluation(model: torch.nn.Module, dataloader: Any, device: torch.device, 
                          logger: RefraktLogger) -> Dict[str, Any]:
    """Run manual evaluation when trainer's evaluate method is not available."""
    model.eval()
    eval_results = {}
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = _extract_batch_data(batch, logger)
            if inputs is None:
                continue
            
            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)
            
            outputs = model(inputs)
            logits = _extract_logits(outputs, logger)
            if logits is None:
                continue
            
            if targets is not None:
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
    
    if total > 0:
        accuracy = correct / total
        eval_results['accuracy'] = accuracy
        logger.info(f"Manual evaluation - Accuracy: {accuracy:.4f}")
    else:
        logger.warning("No valid samples for accuracy calculation")
        eval_results['accuracy'] = None
    
    return eval_results


def _manual_evaluation(model: torch.nn.Module, dataloader: Any, device: torch.device, logger: RefraktLogger) -> Dict[str, Any]:
    """
    Manually evaluate the model when trainer's evaluate method is not available.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        logger: Logger instance
        
    Returns:
        Dict containing evaluation metrics
    """
    return _run_manual_evaluation(model, dataloader, device, logger)
