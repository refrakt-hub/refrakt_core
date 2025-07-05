"""
Inference entry point for Refrakt.

This module orchestrates the inference pipeline using utility functions for config, logger, model, data, and artifact handling.
"""

import gc
import sys
import traceback
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.inference_utils import (
    resolve_model_name_for_inference,
    handle_pure_ml_inference,
    load_fusion_head_if_provided,
    run_inference_loop,
)
from refrakt_core.api.helpers.inference_helpers import (
    _load_and_validate_config,
    _setup_logging,
    _check_pure_ml_inference,
    _setup_device,
    _load_model_and_setup,
    _setup_data_loader,
)

gc.collect()
torch.cuda.empty_cache()

import warnings
warnings.filterwarnings("ignore")


def inference(
    cfg: Union[str, DictConfig],
    model_path: str,
    fusion_head_path: Optional[str] = None,
    data: Any = None,
    logger: Optional[RefraktLogger] = None,
) -> Dict[str, Any]:
    """
    Orchestrate the inference pipeline for Refrakt.

    Args:
        cfg: Path to config file or DictConfig config.
        model_path (str): Path to the model checkpoint.
        fusion_head_path (Optional[str]): Path to the fusion head checkpoint.
        data (Any): Custom data for inference (optional).
        logger (Optional[RefraktLogger]): Logger instance (optional).

    Returns:
        Dict[str, Any]: Inference results and metadata.
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
        fusion_head = load_fusion_head_if_provided(fusion_head_path, logger)

        # Setup data loader
        data_loader = _setup_data_loader(config, data, logger)

        # Setup artifact dumper
        from refrakt_core.api.utils.train_utils import setup_artifact_dumper
        artifact_dumper = setup_artifact_dumper(config, resolved_model_name, logger)

        # Run inference
        results = run_inference_loop(model, data_loader)

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
