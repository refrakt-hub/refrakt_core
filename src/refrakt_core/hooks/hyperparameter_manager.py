"""
Hyperparameter management system for Refrakt with manual override capabilities.

This module provides a flexible system for managing hyperparameters with
support for manual overrides, validation, and type safety.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from omegaconf import DictConfig, OmegaConf


@dataclass
class HyperparameterOverride:
    """Represents a hyperparameter override with validation."""

    path: str  # Dot notation path to the parameter
    value: Any
    description: Optional[str] = None
    validation_func: Optional[Callable[[Any], bool]] = None
    required: bool = False

    def validate(self) -> bool:
        """Validate the override value."""
        if self.validation_func:
            return self.validation_func(self.value)
        return True


class HyperparameterManager:
    """
    Manages hyperparameters with support for manual overrides and validation.

    This class provides a flexible system for managing hyperparameters with
    support for manual overrides, validation, and type safety.
    """

    def __init__(self, base_config: Optional[Union[Dict[str, Any], Any]] = None):
        """
        Initialize the hyperparameter manager.

        Args:
            base_config: Base configuration to start with
        """
        self.base_config = OmegaConf.create(base_config or {})
        self.overrides: List[HyperparameterOverride] = []
        self.validation_rules: Dict[str, Callable[[Any], bool]] = {}
        self._applied_overrides: Set[str] = set()

    def add_override(
        self,
        path: str,
        value: Any,
        description: Optional[str] = None,
        validation_func: Optional[Callable[[Any], bool]] = None,
        required: bool = False,
    ) -> None:
        """
        Add a hyperparameter override.

        Args:
            path: Dot notation path to the parameter (e.g., 'model.params.lr')
            value: New value for the parameter
            description: Optional description of the override
            validation_func: Optional validation function
            required: bool: Whether this override is required
        """
        override = HyperparameterOverride(
            path=path,
            value=value,
            description=description,
            validation_func=validation_func,
            required=required,
        )

        if not override.validate():
            raise ValueError(f"Invalid value for override {path}: {value}")

        self.overrides.append(override)

    def add_validation_rule(
        self, path_pattern: str, validation_func: Callable[[Any], bool]
    ) -> None:
        """
        Add a validation rule for parameters matching a pattern.

        Args:
            path_pattern: Pattern to match parameter paths
            validation_func: Validation function to apply
        """
        self.validation_rules[path_pattern] = validation_func

    def _apply_override(
        self, config: DictConfig, override: HyperparameterOverride
    ) -> None:
        """Apply a single override to the configuration."""
        try:
            # Navigate to the parent of the target
            path_parts = override.path.split(".")
            current = config

            # Navigate to parent
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value
            current[path_parts[-1]] = override.value
            self._applied_overrides.add(override.path)

        except Exception as e:
            raise ValueError(f"Failed to apply override {override.path}: {e}")

    def get_config(self, apply_overrides: bool = True) -> DictConfig:
        """
        Get the current configuration with overrides applied.

        Args:
            apply_overrides: Whether to apply overrides

        Returns:
            Configuration with overrides applied
        """
        config = copy.deepcopy(self.base_config)

        from omegaconf import DictConfig

        if not isinstance(config, DictConfig):
            raise TypeError(
                "Config must be a DictConfig (dictionary-like), not a ListConfig (list-like)."
            )

        if apply_overrides:
            for override in self.overrides:
                self._apply_override(config, override)

        return config

    def get_parameter(self, path: str, default: Any = None) -> Any:
        """
        Get a specific parameter value.

        Args:
            path: Dot notation path to the parameter
            default: Default value if parameter not found

        Returns:
            Parameter value
        """
        config = self.get_config()

        try:
            current = config
            for part in path.split("."):
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default

    def set_parameter(self, path: str, value: Any) -> None:
        """
        Set a parameter value directly.

        Args:
            path: Dot notation path to the parameter
            value: New value for the parameter
        """
        self.add_override(path, value)

    def validate_config(self) -> List[str]:
        """
        Validate the current configuration.

        Returns:
            List of validation errors
        """
        errors = []
        self.get_config()

        # Check required overrides
        for override in self.overrides:
            if override.required and override.path not in self._applied_overrides:
                errors.append(f"Required override not applied: {override.path}")

        # Apply validation rules
        for path_pattern, validation_func in self.validation_rules.items():
            try:
                value = self.get_parameter(path_pattern)
                if value is not None and not validation_func(value):
                    errors.append(f"Validation failed for {path_pattern}")
            except Exception as e:
                errors.append(f"Validation error for {path_pattern}: {e}")

        return errors

    def list_overrides(self) -> List[Dict[str, Any]]:
        """
        List all current overrides.

        Returns:
            List of override information
        """
        return [
            {
                "path": override.path,
                "value": override.value,
                "description": override.description,
                "required": override.required,
                "applied": override.path in self._applied_overrides,
            }
            for override in self.overrides
        ]

    def clear_overrides(self) -> None:
        """Clear all overrides."""
        self.overrides.clear()
        self._applied_overrides.clear()

    def save_config(self, filepath: Union[str, Path]) -> None:
        """
        Save the current configuration to a file.

        Args:
            filepath: Path to save the configuration
        """
        config = self.get_config()
        OmegaConf.save(config, filepath)

    def load_config(self, filepath: Union[str, Path]) -> None:
        """
        Load configuration from a file.

        Args:
            filepath: Path to load the configuration from
        """
        config = OmegaConf.load(filepath)
        from omegaconf import DictConfig

        if not isinstance(config, DictConfig):
            raise TypeError(
                "Loaded config must be a DictConfig (dictionary-like), not a ListConfig (list-like)."
            )
        self.base_config = config


# Predefined validation functions
def validate_positive_float(value: Any) -> bool:
    """Validate that value is a positive float."""
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False


def validate_positive_int(value: Any) -> bool:
    """Validate that value is a positive integer."""
    try:
        return int(value) > 0
    except (ValueError, TypeError):
        return False


def validate_range(value: Any, min_val: float, max_val: float) -> bool:
    """Validate that value is within a range."""
    try:
        val = float(value)
        return min_val <= val <= max_val
    except (ValueError, TypeError):
        return False


def validate_choice(value: Any, choices: List[Any]) -> bool:
    """Validate that value is one of the allowed choices."""
    return value in choices


def validate_device(value: Any) -> bool:
    """Validate that value is a valid device."""
    if isinstance(value, str):
        return value in ["cpu", "cuda", "mps"] or value.startswith("cuda:")
    return False


def validate_model_type(value: Any) -> bool:
    """Validate that value is a valid model type."""
    valid_types = [
        "resnet",
        "convnext",
        "swin",
        "vit",
        "autoencoder",
        "dino",
        "mae",
        "msn",
        "simclr",
        "srgan",
        "gan",
    ]
    return value in valid_types


# Convenience functions for common hyperparameter patterns
def create_learning_rate_override(
    value: float, description: Optional[str] = None
) -> HyperparameterOverride:
    """Create a learning rate override with validation."""
    return HyperparameterOverride(
        path="training.optimizer.params.lr",
        value=value,
        description=description or f"Learning rate override: {value}",
        validation_func=validate_positive_float,
        required=False,
    )


def create_batch_size_override(
    value: int, description: Optional[str] = None
) -> HyperparameterOverride:
    """Create a batch size override with validation."""
    return HyperparameterOverride(
        path="training.dataloader.params.batch_size",
        value=value,
        description=description or f"Batch size override: {value}",
        validation_func=validate_positive_int,
        required=False,
    )


def create_epochs_override(
    value: int, description: Optional[str] = None
) -> HyperparameterOverride:
    """Create an epochs override with validation."""
    return HyperparameterOverride(
        path="training.params.epochs",
        value=value,
        description=description or f"Epochs override: {value}",
        validation_func=validate_positive_int,
        required=False,
    )


def create_device_override(
    value: str, description: Optional[str] = None
) -> HyperparameterOverride:
    """Create a device override with validation."""
    return HyperparameterOverride(
        path="training.params.device",
        value=value,
        description=description or f"Device override: {value}",
        validation_func=validate_device,
        required=False,
    )
