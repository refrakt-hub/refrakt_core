"""
Helper functions for the test API.

This module contains internal helper functions used by the main test function.
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
from refrakt_core.api.utils.test_utils import (
    _resolve_model_name,
    _handle_pure_ml_pipeline,
    _setup_fusion_evaluation,
    _run_manual_evaluation,
    _load_model_checkpoint,
    _build_test_loader_with_resize,
)


def _load_and_validate_config(cfg: Union[str, DictConfig]) -> DictConfig:
    """Load and validate configuration."""
    if isinstance(cfg, str):
        config = load_config(cfg)
    else:
        config = cfg
    return config


def _setup_logging(config: DictConfig, resolved_model_name: str, logger: Optional[RefraktLogger]) -> RefraktLogger:
    """Setup logging configuration."""
    if logger is None:
        logger = setup_logger(config, resolved_model_name)
    
    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a dict after OmegaConf.to_container.")
    logger.log_config(cast(Dict[str, Any], config_dict))
    return logger


def _check_pure_ml_testing(config: DictConfig) -> bool:
    """Check if this is a pure ML testing session."""
    return (getattr(config.model, 'type', None) == 'ml' or 
            getattr(config.dataset, 'name', None) == 'tabular_ml')


def _get_modules_and_device() -> tuple[Dict[str, Any], torch.device]:
    """Get registry modules and device."""
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
    return modules, device


def _build_test_components(config: DictConfig, modules: Dict[str, Any], device: torch.device, logger: RefraktLogger):
    """Build test components (dataloader, model, loss)."""
    from refrakt_core.api.builders.model_builder import build_model
    from refrakt_core.api.builders.loss_builder import build_loss

    dataloader = _build_test_loader_with_resize(config, logger)
    model_cls = modules["get_model"](config.model.name)

    model = build_model(
        cast(OmegaConf, config),
        modules={
            "get_model": modules["get_model"],
            "get_wrapper": modules["get_wrapper"],
            "model": model_cls,
        },
        device=str(device),
    )

    loss_fn = build_loss(
        cast(OmegaConf, config), modules=modules, device=str(device)
    )

    return dataloader, model, loss_fn


def _setup_trainer_for_testing(config: DictConfig, model: torch.nn.Module, dataloader: Any, 
                              loss_fn: Any, device: str, modules: Dict[str, Any], 
                              artifact_dumper: Any, resolved_model_name: str, logger: RefraktLogger):
    """Setup trainer for testing."""
    from refrakt_core.api.builders.trainer_builder import initialize_trainer

    trainer = initialize_trainer(
        cfg=cast(OmegaConf, config),
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        loss_fn=loss_fn,
        optimizer=None,
        scheduler=None,
        device=device,
        modules=modules,
        save_dir=None,
    )
    trainer.model_name = resolved_model_name
    trainer.logger = logger
    trainer.artifact_dumper = artifact_dumper

    return trainer


def _evaluate_model(trainer: Any, model: torch.nn.Module, dataloader: Any, device: torch.device, 
                   fusion_acc: Optional[float], logger: RefraktLogger) -> Dict[str, Any]:
    """Evaluate model performance."""
    model.eval()
    eval_results = {}
    
    # Use trainer's evaluate method if available
    if hasattr(trainer, 'evaluate'):
        try:
            if fusion_acc is not None:
                # For fusion models, we already evaluated above
                eval_results['fusion_accuracy'] = fusion_acc
            else:
                # For regular models, use trainer's evaluate method
                accuracy = trainer.evaluate()
                eval_results['accuracy'] = accuracy
                logger.info(f"Model accuracy: {accuracy:.4f}")
        except Exception as e:
            logger.warning(f"Could not use trainer's evaluate method: {e}")
            eval_results = _run_manual_evaluation(model, dataloader, device, logger)
    else:
        # Manual evaluation if trainer doesn't have evaluate method
        eval_results = _run_manual_evaluation(model, dataloader, device, logger)

    return eval_results 