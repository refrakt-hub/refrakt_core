"""
CLI helper functions for Refrakt.

This module contains internal helper functions used by the main CLI entry point.
"""

import argparse
from typing import Any, List, Optional, cast

from omegaconf import DictConfig, OmegaConf


def _setup_argument_parser() -> argparse.ArgumentParser:
    """Setup argument parser for CLI."""
    parser = argparse.ArgumentParser(description="Refrakt Core Pipeline")
    parser.add_argument(
        "--config", required=True, help="Path to configuration file"
    )
    parser.add_argument("--log_dir", help="Override log directory path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--override", 
        nargs='+',
        help="Override configuration values (format: path.to.param=value). Can specify multiple overrides."
    )
    return parser


def _extract_overrides(args, remaining) -> List[str]:
    """Extract and combine all overrides from arguments."""
    from refrakt_core.hooks.hyperparameter_override import extract_overrides_from_args
    positional_overrides, _ = extract_overrides_from_args(remaining)
    
    # Combine explicit --override flags with positional overrides
    all_overrides = []
    if args.override:
        all_overrides.extend(args.override)
    all_overrides.extend(positional_overrides)
    
    return all_overrides


def _apply_config_overrides(cfg: Any, all_overrides: List[str]) -> Any:
    """Apply overrides to configuration."""
    if all_overrides:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg_dict, dict):
            print(f"DEBUG: Before overrides - batch_size: {cfg_dict.get('dataloader', {}).get('params', {}).get('batch_size', 'NOT_FOUND')}")
            from refrakt_core.hooks.hyperparameter_override import apply_overrides
            cfg_dict = apply_overrides(OmegaConf.create(cfg_dict), all_overrides)
            cfg = OmegaConf.create(cfg_dict)
            print(f"DEBUG: After overrides - batch_size: {cfg_dict.get('dataloader', {}).get('params', {}).get('batch_size', 'NOT_FOUND')}")
    return cfg


def _extract_runtime_config(cfg: DictConfig) -> dict:
    """Extract runtime configuration from config."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(
            "Config must be a dict after OmegaConf.to_container, got type: {}".format(
                type(cfg_dict)
            )
        )
    cfg_dict = cast(dict, cfg_dict)
    return cfg_dict.get("runtime", {})


def _setup_logging_config(runtime_cfg: dict, args_log_dir: Optional[str] = None) -> tuple:
    """Setup logging configuration from runtime config."""
    mode = runtime_cfg.get("mode", "train")
    log_dir = args_log_dir or runtime_cfg.get("log_dir", "./logs")

    # Handle log_types - accept list or single string
    log_types = runtime_cfg.get("log_type", [])
    if isinstance(log_types, str):
        log_types = [log_types]  # Convert single string to list
    elif log_types is None:
        log_types = []  # Convert None to empty list

    console = runtime_cfg.get("console", True)
    model_path = runtime_cfg.get("model_path", None)
    debug = runtime_cfg.get("debug", False)

    return mode, log_dir, log_types, console, model_path, debug


def _execute_pipeline_mode(mode: str, cfg: DictConfig, model_path: str, logger) -> None:
    """Execute the appropriate pipeline based on mode."""
    from refrakt_core.api.utils.pipeline_utils import (
        execute_training_pipeline,
        execute_testing_pipeline,
        execute_inference_pipeline,
        execute_full_pipeline,
    )

    if mode == "train":
        execute_training_pipeline(cfg, model_path, logger)

    elif mode == "test":
        execute_testing_pipeline(cfg, model_path, logger)

    elif mode == "inference":
        if not model_path:
            raise ValueError(
                "model_path must be provided in runtime config for inference mode"
            )
        execute_inference_pipeline(cfg, model_path, logger)

    elif mode == "pipeline":
        execute_full_pipeline(cfg, logger) 