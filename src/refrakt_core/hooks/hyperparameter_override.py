"""
Hyperparameter override system for Refrakt with command-line support.

This module provides a flexible system for managing hyperparameters with
support for command-line overrides and validation.
"""

import copy
import re
from pathlib import Path
from typing import Any, List, Optional, Union

from omegaconf import DictConfig, ListConfig, OmegaConf


def parse_override_arg(override_arg: str) -> tuple[str, Any]:
    """
    Parse a command-line override argument.

    Args:
        override_arg: Override argument in format 'path.to.param=value'

    Returns:
        Tuple of (path, value)

    Raises:
        ValueError: If override format is invalid
    """
    if "=" not in override_arg:
        raise ValueError(
            f"Invalid override format: {override_arg}. Expected 'path.to.param=value'"
        )

    path, value_str = override_arg.split("=", 1)
    path = path.strip()
    value_str = value_str.strip()

    # Try to convert value to appropriate type
    value = _convert_value(value_str)

    return path, value


def _convert_value(value_str: str) -> Any:
    """
    Convert string value to appropriate type.

    Args:
        value_str: String value to convert

    Returns:
        Converted value
    """
    # Try boolean
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"

    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def apply_overrides(config: DictConfig, overrides: List[str]) -> DictConfig:
    """
    Apply command-line overrides to a configuration.

    Args:
        config: Base configuration
        overrides: List of override strings in format 'path.to.param=value'

    Returns:
        Configuration with overrides applied
    """
    if not isinstance(config, DictConfig):
        raise TypeError("apply_overrides only supports DictConfig, not ListConfig")
    config_copy = copy.deepcopy(config)
    for override in overrides:
        try:
            path, value = parse_override_arg(override)
            _set_nested_value(config_copy, path, value)
        except Exception as e:
            raise ValueError(f"Failed to apply override '{override}': {e}")
    return config_copy


def _set_nested_value(config: DictConfig, path: str, value: Any) -> None:
    """
    Set a nested value in configuration using dot notation.

    Args:
        config: Configuration to modify
        path: Dot notation path (e.g., 'model.params.lr')
        value: Value to set
    """
    path_parts = path.split(".")
    current = config

    # Navigate to parent
    for part in path_parts[:-1]:
        if part not in current or current[part] is None:
            current[part] = OmegaConf.create({})
        current = current[part]

    # Set the value
    current[path_parts[-1]] = value


def validate_overrides(overrides: List[str]) -> List[str]:
    """
    Validate override arguments.

    Args:
        overrides: List of override strings

    Returns:
        List of validation errors
    """
    errors = []

    for i, override in enumerate(overrides):
        try:
            path, value = parse_override_arg(override)

            # Basic path validation
            if not path or path.startswith(".") or path.endswith("."):
                errors.append(f"Invalid path in override {i+1}: '{path}'")

            # Check for invalid characters in path
            if re.search(r"[^a-zA-Z0-9._]", path):
                errors.append(f"Invalid characters in path '{path}' in override {i+1}")

        except ValueError as e:
            errors.append(f"Override {i+1}: {e}")

    return errors


def create_override_help() -> str:
    """
    Create help text for override usage.

    Returns:
        Help text string
    """
    return """
Hyperparameter Overrides:
    Use the format 'path.to.parameter=value' to override configuration values.
    
    Examples:
        config.model.name=ResNet
        config.optimizer.params.lr=0.0005
        config.trainer.params.epochs=20
        config.data.params.batch_size=32
        config.device=cuda
    
    Supported value types:
        - Strings: config.model.name=ResNet
        - Numbers: config.optimizer.lr=0.001
        - Booleans: config.debug=true
        - Integers: config.epochs=100
    """


def extract_overrides_from_args(args: List[str]) -> tuple[List[str], List[str]]:
    """
    Extract override arguments from command-line arguments.

    Args:
        args: List of command-line arguments

    Returns:
        Tuple of (overrides, remaining_args)
    """
    overrides = []
    remaining_args = []

    for arg in args:
        if "=" in arg and not arg.startswith("-"):
            overrides.append(arg)
        else:
            remaining_args.append(arg)

    return overrides, remaining_args


def load_config_with_overrides(
    config_path: Union[str, Path], overrides: Optional[List[str]] = None
) -> DictConfig:
    """
    Load configuration from file and apply overrides.

    Args:
        config_path: Path to configuration file
        overrides: Optional list of override strings

    Returns:
        Configuration with overrides applied
    """
    # Load base configuration
    config = OmegaConf.load(config_path)
    if isinstance(config, ListConfig):
        raise TypeError("Config file must yield a DictConfig, not ListConfig")
    # Apply overrides if provided
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def save_config_with_overrides(
    config: DictConfig,
    output_path: Union[str, Path],
    overrides: Optional[List[str]] = None,
) -> None:
    """
    Save configuration with overrides to file.

    Args:
        config: Base configuration
        output_path: Path to save configuration
        overrides: Optional list of override strings
    """
    if overrides:
        config = apply_overrides(config, overrides)

    OmegaConf.save(config, output_path)


def get_parameter_value(config: DictConfig, path: str, default: Any = None) -> Any:
    """
    Get a parameter value from configuration using dot notation.

    Args:
        config: Configuration to search
        path: Dot notation path
        default: Default value if parameter not found

    Returns:
        Parameter value or default
    """
    try:
        current = config
        for part in path.split("."):
            current = current[part]
        return current
    except (KeyError, TypeError):
        return default


def set_parameter_value(config: DictConfig, path: str, value: Any) -> None:
    """
    Set a parameter value in configuration using dot notation.

    Args:
        config: Configuration to modify
        path: Dot notation path
        value: Value to set
    """
    _set_nested_value(config, path, value)


def validate_config_structure(
    config: DictConfig, required_paths: List[str]
) -> List[str]:
    """
    Validate that required configuration paths exist.

    Args:
        config: Configuration to validate
        required_paths: List of required paths

    Returns:
        List of validation errors
    """
    errors = []

    for path in required_paths:
        value = get_parameter_value(config, path)
        if value is None:
            errors.append(f"Required configuration path not found: {path}")

    return errors


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations with override_config taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = copy.deepcopy(base_config)

    def merge_dicts(base: Any, override: Any) -> None:
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], (dict, DictConfig))
                and isinstance(value, (dict, DictConfig))
            ):
                merge_dicts(base[key], value)
            else:
                base[key] = value

    merge_dicts(merged, override_config)
    return merged
