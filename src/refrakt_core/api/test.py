"""
Test entry point for Refrakt.

This module orchestrates the test pipeline using utility functions for config, logger, model, data, and artifact handling.
"""

import gc
import os
import sys
import traceback
from typing import Any, Dict, Optional, cast

import torch
from omegaconf import OmegaConf
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
    cfg: Any, model_path: Optional[str] = None, logger: Optional[RefraktLogger] = None
) -> None:
    """
    Orchestrate the test pipeline for Refrakt.

    Args:
        cfg: Path to config file or OmegaConf config.
        model_path (Optional[str]): Path to the model checkpoint.
        logger (Optional[RefraktLogger]): Logger instance (optional).
    """
    try:
        config = load_config(cfg)
        # Resolve model name
        if config.model.name == "autoencoder":
            variant = config.model.params.get("variant", "simple")
            resolved_model_name = f"autoencoder_{variant}"
        else:
            resolved_model_name = config.model.name

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
        dataloader = _build_test_loader(config)
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

        # Log model outputs per batch (optional, can be expanded as needed)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if isinstance(batch, torch.Tensor):
                    inputs = batch
                elif isinstance(batch, dict):
                    inputs = batch.get("input") or batch.get("image") or batch.get("lr")
                    if inputs is None:
                        raise ValueError("No valid input key found in batch.")
                else:
                    continue
                outputs = model(inputs)
                # Optionally log outputs here

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
