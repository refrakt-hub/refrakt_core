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

warnings.filterwarnings("ignore")

gc.collect()
torch.cuda.empty_cache()


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


inference._load_and_validate_config = _load_and_validate_config

__all__ = ["inference", "_load_and_validate_config"]
