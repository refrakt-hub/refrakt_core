"""
Inference entry point for Refrakt.

This module orchestrates the inference pipeline using utility functions for config, logger, model, data, and artifact handling.
"""

import gc
import os
import sys
import traceback
from typing import Any, Dict, Optional, Union, cast

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.train_utils import (
    load_config, load_fusion_head, setup_artifact_dumper,
    setup_data_loader_for_inference, setup_logger)
from refrakt_core.global_logging import get_global_logger

gc.collect()
torch.cuda.empty_cache()

import warnings

warnings.filterwarnings("ignore")


def inference(
    cfg: Union[str, OmegaConf],
    model_path: str,
    fusion_head_path: Optional[str] = None,
    data: Any = None,
    logger: Optional[RefraktLogger] = None,
) -> Dict[str, Any]:
    """
    Orchestrate the inference pipeline for Refrakt.

    Args:
        cfg: Path to config file or OmegaConf config.
        model_path (str): Path to the model checkpoint.
        fusion_head_path (Optional[str]): Path to the fusion head checkpoint.
        data (Any): Custom data for inference (optional).
        logger (Optional[RefraktLogger]): Logger instance (optional).

    Returns:
        Dict[str, Any]: Inference results and metadata.
    """
    try:
        # Ensure cfg is str or DictConfig for load_config
        config = load_config(cast("str | DictConfig", cfg))
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
        logger.log_config(cast(Dict[str, Any], config_dict))

        # Modules and device
        from refrakt_core.registry.model_registry import get_model
        from refrakt_core.registry.wrapper_registry import get_wrapper

        modules = {
            "get_model": get_model,
            "get_wrapper": get_wrapper,
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        from refrakt_core.api.builders.model_builder import build_model

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

        # Load checkpoint
        from refrakt_core.api.utils.test_utils import _load_model_checkpoint

        _load_model_checkpoint(model, model_path, device, logger)
        model.eval()

        # Load fusion head if provided
        fusion_head = None
        if fusion_head_path and os.path.exists(fusion_head_path):
            fusion_head = load_fusion_head(fusion_head_path)
            logger.info(f"Loaded fusion head from {fusion_head_path}")

        # Data loader
        data_loader = setup_data_loader_for_inference(config, data)

        # Artifact Dumper
        artifact_dumper = setup_artifact_dumper(config, resolved_model_name, logger)

        # Inference loop
        results = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if isinstance(batch, torch.Tensor):
                    inputs = batch
                elif isinstance(batch, dict):
                    inputs = batch.get("input") or batch.get("image") or batch.get("lr")
                    if inputs is None:
                        raise ValueError("No valid input key found in batch.")
                else:
                    continue
                outputs = model(inputs)
                results.append(outputs)
                # Optionally log outputs here

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
