"""
Registry system for Refrakt - safe drop-in replacement.

This module provides a thread-safe, import-safe registry system that can
be used as a drop-in replacement for the existing registry system.
"""

from typing import Any, Iterator, List, Optional

from refrakt_core.registry.safe_registry import (  # Safe registry functions; Drop-in replacement functions
    clear_registry,
    get_component,
    get_dataset,
    get_loss,
    get_model,
    get_registry,
    get_trainer,
    get_transform,
    list_components,
    register_component,
    register_dataset,
    register_loss,
    register_model,
    register_trainer,
    register_transform,
)

# Backward compatibility - export the same names as the old registry
__all__ = [
    # Safe registry functions
    "get_registry",
    "register_component",
    "get_component",
    "list_components",
    "clear_registry",
    # Drop-in replacement functions
    "register_dataset",
    "get_dataset",
    "register_model",
    "get_model",
    "register_loss",
    "get_loss",
    "register_trainer",
    "get_trainer",
    "register_transform",
    "get_transform",
    # Legacy compatibility (these will be removed in future versions)
    "DATASET_REGISTRY",
    "MODEL_REGISTRY",
    "LOSS_REGISTRY",
    "TRAINER_REGISTRY",
    "TRANSFORM_REGISTRY",
]


# Legacy compatibility - create registry objects that forward to the safe registry
class LegacyRegistry:
    """Legacy registry object for backward compatibility."""

    def __init__(self, registry_name: str) -> None:
        self.registry_name = registry_name

    def __getitem__(self, key: str) -> Any:
        return get_component(self.registry_name, key)

    def __setitem__(self, key: str, value: Any) -> Any:
        register_component(self.registry_name, key)(value)
        return value

    def __contains__(self, key: str) -> bool:
        try:
            get_component(self.registry_name, key)
            return True
        except ValueError:
            return False

    def keys(self) -> List[str]:
        return list_components(self.registry_name)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return get_component(self.registry_name, key, default=default)


# Create legacy registry objects
DATASET_REGISTRY = LegacyRegistry("datasets")
MODEL_REGISTRY = LegacyRegistry("models")
LOSS_REGISTRY = LegacyRegistry("losses")
TRAINER_REGISTRY = LegacyRegistry("trainers")
TRANSFORM_REGISTRY = LegacyRegistry("transforms")
