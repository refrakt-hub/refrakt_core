"""
CLI helper functions for Refrakt.

This module provides internal helper functions used by the main CLI entry point
to handle argument parsing, configuration overrides, and pipeline execution.

The module handles:
- Command-line argument parsing and validation
- Configuration override extraction and application
- Runtime configuration setup and validation
- Logging configuration management
- Pipeline mode execution routing

These utilities ensure robust CLI operation with proper error handling
and configuration management for different pipeline modes.

Typical usage involves calling these helper functions from the main CLI
entry point to set up and execute the appropriate pipeline.
"""

import argparse
from typing import Any, Dict, List, Optional, Tuple, cast

from omegaconf import DictConfig, OmegaConf


def _setup_argument_parser() -> argparse.ArgumentParser:
    """
    Setup argument parser for CLI with all required arguments.

    This function creates and configures the argument parser with all
    necessary arguments for the Refrakt CLI, including configuration
    file path, logging options, and override capabilities.

    Returns:
        Configured argument parser ready for CLI usage
    """
    parser = argparse.ArgumentParser(description="Refrakt Core Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--log_dir", help="Override log directory path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--override",
        nargs="+",
        help="Specify multiple override values (format: path.to.param=value).",
    )
    return parser


def _extract_overrides(args: argparse.Namespace, remaining: List[str]) -> List[str]:
    """
    Extract and combine all overrides from arguments.

    This function combines explicit --override flags with positional overrides
    to create a comprehensive list of configuration overrides to apply.

    Args:
        args: Parsed command-line arguments
        remaining: Remaining positional arguments that may contain overrides

    Returns:
        Combined list of all override strings to apply

    Note:
        This function handles both explicit --override flags and positional
        overrides, ensuring all override methods are properly combined.
    """
    from refrakt_core.hooks.hyperparameter_override import extract_overrides_from_args

    positional_overrides, _ = extract_overrides_from_args(remaining)

    # Combine explicit --override flags with positional overrides
    all_overrides: List[str] = []
    if args.override:
        all_overrides.extend(args.override)
    all_overrides.extend(positional_overrides)

    return all_overrides


def _apply_config_overrides(cfg: Any, all_overrides: List[str]) -> Any:
    """
    Apply overrides to configuration.

    This function applies configuration overrides to the main configuration
    object, enabling runtime parameter modifications without changing
    configuration files.

    Args:
        cfg: Configuration object to modify
        all_overrides: List of override strings to apply

    Returns:
        Updated configuration object with overrides applied

    Note:
        The function includes debug logging to track override application
        and ensure proper parameter modification.
    """
    if all_overrides:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg_dict, dict):
            from refrakt_core.hooks.hyperparameter_override import apply_overrides

            cfg_dict = apply_overrides(OmegaConf.create(cfg_dict), all_overrides)
            cfg = OmegaConf.create(cfg_dict)
    return cfg


def _extract_runtime_config(cfg: DictConfig) -> Dict[str, Any]:
    """
    Extract runtime configuration from config.

    This function extracts the runtime configuration section from the main
    configuration object, which contains pipeline execution settings.

    Args:
        cfg: Configuration object containing runtime settings

    Returns:
        Runtime configuration dictionary

    Raises:
        TypeError: If the configuration cannot be converted to a dictionary
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(
            "Config must be a dict after OmegaConf.to_container, got type: {}".format(
                type(cfg_dict)
            )
        )
    cfg_dict = cast(Dict[str, Any], cfg_dict)
    runtime_config = cfg_dict.get("runtime", {})
    if isinstance(runtime_config, dict):
        return runtime_config
    else:
        return {}


def _setup_logging_config(
    runtime_cfg: Dict[str, Any], args_log_dir: Optional[str] = None
) -> Tuple[str, str, List[str], bool, Optional[str], bool]:
    """
    Setup logging configuration from runtime config.

    This function extracts and validates logging configuration parameters
    from the runtime configuration, handling various parameter types and
    providing sensible defaults.

    Args:
        runtime_cfg: Runtime configuration dictionary
        args_log_dir: Optional log directory override from command line

    Returns:
        Tuple containing:
        - mode: Pipeline execution mode
        - log_dir: Log directory path
        - log_types: List of logging backends
        - console: Whether to enable console logging
        - model_path: Optional model path for inference
        - debug: Whether debug logging is enabled
    """
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


def _execute_pipeline_mode(
    mode: str, cfg: DictConfig, model_path: str, logger: Any
) -> None:
    """
    Execute the appropriate pipeline based on mode.

    This function routes to the correct pipeline execution function based
    on the specified mode, handling different pipeline types with appropriate
    validation and error handling.

    Args:
        mode: Pipeline execution mode ('train', 'test', 'inference', 'pipeline')
        cfg: Configuration object for the pipeline
        model_path: Model path for inference mode
        logger: Logger instance for status messages

    Raises:
        ValueError: If model_path is required but not provided for inference mode
    """
    from refrakt_core.api.utils.pipeline_utils import (
        execute_full_pipeline,
        execute_inference_pipeline,
        execute_testing_pipeline,
        execute_training_pipeline,
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
