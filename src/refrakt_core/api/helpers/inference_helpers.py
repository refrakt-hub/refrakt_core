"""
Helper functions for the inference API.

This module contains internal helper functions used by the main inference function.
"""

import torch
from typing import Any, Dict, Optional, cast, Union
from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.train_utils import (
    load_config,
    setup_logger,
    setup_artifact_dumper,
)
from refrakt_core.api.utils.inference_utils import (
    resolve_model_name_for_inference,
    handle_pure_ml_inference,
    setup_model_and_device,
    load_fusion_head_if_provided,
    run_inference_loop,
)


def _load_and_validate_config(cfg: Union[str, DictConfig]) -> DictConfig:
    """Load and validate configuration."""
    return load_config(cast("str | DictConfig", cfg))


def _setup_logging(config: DictConfig, resolved_model_name: str, logger: Optional[RefraktLogger]) -> RefraktLogger:
    """Setup logging configuration."""
    if logger is None:
        logger = setup_logger(config, resolved_model_name)
    
    config_dict = OmegaConf.to_container(config, resolve=True)
    logger.log_config(cast(Dict[str, Any], config_dict))
    return logger


def _check_pure_ml_inference(config: DictConfig) -> bool:
    """Check if this is a pure ML inference session."""
    return (getattr(config.model, 'type', None) == 'ml' or 
            getattr(config.dataset, 'name', None) == 'tabular_ml')


def _setup_device() -> torch.device:
    """Setup device for inference."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model_and_setup(config: DictConfig, device: torch.device, model_path: str, logger: RefraktLogger):
    """Load model and setup for inference."""
    from refrakt_core.api.utils.test_utils import _load_model_checkpoint
    
    model, modules = setup_model_and_device(config, device)
    _load_model_checkpoint(model, model_path, device, logger)
    model.eval()
    
    return model, modules


def _setup_data_loader(config: DictConfig, data: Any, logger: RefraktLogger):
    """Setup data loader for inference."""
    from refrakt_core.api.utils.train_utils import setup_data_loader_for_inference_with_resize
    return setup_data_loader_for_inference_with_resize(config, data, logger) 