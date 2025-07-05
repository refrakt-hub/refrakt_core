"""
Test entry point for Refrakt.

This module orchestrates the test pipeline using utility functions for config, logger, model, data, and artifact handling.
"""

import gc
import os
import sys
import traceback
from typing import Any, Dict, Optional, cast, Union

import torch
import torch.nn
from omegaconf import OmegaConf, DictConfig
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.test_utils import (_build_test_loader,
                                               _load_model_checkpoint)
from refrakt_core.api.utils.train_utils import (load_config, load_fusion_head,
                                                setup_artifact_dumper,
                                                setup_logger)
from refrakt_core.global_logging import get_global_logger
from refrakt_core.integrations.gpu.wrapper import CuMLWrapper
from refrakt_core.integrations.fusion.builder import build_fusion_head
from refrakt_core.integrations.fusion.trainer import FusionTrainer
from refrakt_core.integrations.cpu.wrapper import SklearnWrapper

gc.collect()
torch.cuda.empty_cache()

import warnings

warnings.filterwarnings("ignore")


def test(
    cfg: Union[str, DictConfig], model_path: Optional[str] = None, logger: Optional[RefraktLogger] = None
) -> None:
    """
    Orchestrate the test pipeline for Refrakt.

    Args:
        cfg (Union[str, DictConfig]): Path to config file or DictConfig object.
        model_path (Optional[str]): Path to the model checkpoint.
        logger (Optional[RefraktLogger]): Logger instance (optional).
    """
    try:
        # Load config if it's a string, otherwise use as-is
        if isinstance(cfg, str):
            config = load_config(cfg)
        else:
            config = cfg
        # Resolve model name
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

        # Logger
        if logger is None:
            logger = setup_logger(config, resolved_model_name)
        config_dict = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config_dict, dict):
            raise TypeError("Config must be a dict after OmegaConf.to_container.")
        logger.log_config(cast(Dict[str, Any], config_dict))

        # --- PURE-ML PIPELINE SUPPORT ---
        is_pure_ml = getattr(config.model, 'type', None) == 'ml' or getattr(config.dataset, 'name', None) == 'tabular_ml'
        if is_pure_ml:
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
            return

        # Modules and device
        from refrakt_core.registry.loss_registry import get_loss
        from refrakt_core.registry.model_registry import get_model
        from refrakt_core.registry.trainer_registry import get_trainer
        from refrakt_core.registry.wrapper_registry import get_wrapper

        modules = {
            "get_model": get_model,
            "get_loss": get_loss,
            "get_trainer": get_trainer,
            "get_wrapper": get_wrapper,
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data, Model, Loss
        from refrakt_core.api.utils.test_utils import _build_test_loader_with_resize
        dataloader = _build_test_loader_with_resize(config, logger)
        model_cls = get_model(config.model.name)
        from refrakt_core.api.builders.model_builder import build_model

        model = build_model(
            cast(OmegaConf, config),
            modules={
                "get_model": get_model,
                "get_wrapper": get_wrapper,
                "model": model_cls,
            },
            device=str(device),
        )
        from refrakt_core.api.builders.loss_builder import build_loss

        loss_fn = build_loss(
            cast(OmegaConf, config), modules=modules, device=str(device)
        )

        # Artifact Dumper
        artifact_dumper = setup_artifact_dumper(config, resolved_model_name, logger)

        # Trainer
        from refrakt_core.api.builders.trainer_builder import \
            initialize_trainer

        trainer = initialize_trainer(
            cfg=cast(OmegaConf, config),
            model=model,
            train_loader=dataloader,
            val_loader=dataloader,
            loss_fn=loss_fn,
            optimizer=None,
            scheduler=None,
            device=str(device),
            modules=modules,
            save_dir=None,
        )
        trainer.model_name = resolved_model_name
        trainer.logger = logger
        trainer.artifact_dumper = artifact_dumper

        # Load Checkpoint
        _load_model_checkpoint(model, model_path, device, logger)

        # Fusion Head
        fusion_cfg = getattr(config.model, "fusion", None)
        fusion_acc = None
        if fusion_cfg:
            fusion_type = fusion_cfg.type
            fusion_model_key = fusion_cfg.model
            fusion_model_path = os.path.join(
                config.trainer.params.save_dir, f"{config.model.name}_fusion.joblib"
            )
            if os.path.exists(fusion_model_path):
                logger.info(f"[FUSION] Found fusion head at {fusion_model_path}")
                if fusion_type == "sklearn":
                    fusion_head = SklearnWrapper.load(
                        fusion_model_key, fusion_model_path
                    )
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
                logger.info(
                    f"[FUSION] Validation accuracy (fusion head): {fusion_acc:.4f}"
                )
            else:
                logger.warning(
                    f"[FUSION] No fusion model found at: {fusion_model_path}"
                )

        # Evaluate model performance
        model.eval()
        eval_results = {}
        
        # Use trainer's evaluate method if available
        if hasattr(trainer, 'evaluate'):
            try:
                if fusion_cfg and fusion_acc is not None:
                    # For fusion models, we already evaluated above
                    eval_results['fusion_accuracy'] = fusion_acc
                else:
                    # For regular models, use trainer's evaluate method
                    accuracy = trainer.evaluate()
                    eval_results['accuracy'] = accuracy
                    logger.info(f"Model accuracy: {accuracy:.4f}")
            except Exception as e:
                logger.warning(f"Could not use trainer's evaluate method: {e}")
                # Fall back to manual evaluation (inline)
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in dataloader:
                        if isinstance(batch, torch.Tensor):
                            inputs = batch
                            targets = None
                        elif isinstance(batch, dict):
                            inputs = batch.get("input") or batch.get("image") or batch.get("lr")
                            targets = batch.get("target") or batch.get("label")
                            if inputs is None:
                                logger.warning("No valid input key found in batch, skipping...")
                                continue
                        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            inputs, targets = batch[0], batch[1]
                        else:
                            logger.warning(f"Unexpected batch format: {type(batch)}, skipping...")
                            continue
                        inputs = inputs.to(device)
                        if targets is not None:
                            targets = targets.to(device)
                        outputs = model(inputs)
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        elif isinstance(outputs, torch.Tensor):
                            logits = outputs
                        else:
                            logger.warning("Could not extract logits from model output")
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
        else:
            # Manual evaluation if trainer doesn't have evaluate method (inline)
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, torch.Tensor):
                        inputs = batch
                        targets = None
                    elif isinstance(batch, dict):
                        inputs = batch.get("input") or batch.get("image") or batch.get("lr")
                        targets = batch.get("target") or batch.get("label")
                        if inputs is None:
                            logger.warning("No valid input key found in batch, skipping...")
                            continue
                    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                    else:
                        logger.warning(f"Unexpected batch format: {type(batch)}, skipping...")
                        continue
                    inputs = inputs.to(device)
                    if targets is not None:
                        targets = targets.to(device)
                    outputs = model(inputs)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, torch.Tensor):
                        logits = outputs
                    else:
                        logger.warning("Could not extract logits from model output")
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

        logger.info("\n✅ Testing completed successfully!")
        print("\nEvaluation Results:", eval_results)

    except Exception as e:
        if not (logger and hasattr(logger, "error")):
            logger = None
        if logger:
            logger.error(f"\n❌ Testing failed: {str(e)}")
            logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        gc.collect()
        torch.cuda.empty_cache()


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
    model.eval()
    eval_results = {}
    
    try:
        # Try to compute accuracy for classification tasks
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, torch.Tensor):
                    inputs = batch
                    targets = None  # No targets provided
                elif isinstance(batch, dict):
                    inputs = batch.get("input") or batch.get("image") or batch.get("lr")
                    targets = batch.get("target") or batch.get("label")
                    if inputs is None:
                        logger.warning("No valid input key found in batch, skipping...")
                        continue
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    logger.warning(f"Unexpected batch format: {type(batch)}, skipping...")
                    continue
                
                inputs = inputs.to(device)
                if targets is not None:
                    targets = targets.to(device)
                
                outputs = model(inputs)
                
                # Extract logits for classification
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logger.warning("Could not extract logits from model output")
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
            
    except Exception as e:
        logger.warning(f"Manual evaluation failed: {e}")
        eval_results['error'] = str(e)
    
    return eval_results
