"""
Safe registry system for Refrakt - drop-in replacement for existing registry.

This module provides a thread-safe, import-safe registry system that can
be used as a drop-in replacement for the existing registry system.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class SafeRegistry:
    """
    Thread-safe registry that avoids global variables and provides safe imports.

    This registry uses a singleton pattern with thread safety and provides
    methods for safe registration and retrieval of components.
    """

    _instance: Optional["SafeRegistry"] = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls) -> "SafeRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._registries: Dict[str, Dict[str, Any]] = {}
        self._import_callbacks: Dict[str, Callable[[], None]] = {}
        self._logger = logging.getLogger(f"{__name__}.SafeRegistry")
        self._initialized = True

    def register_import_callback(
        self, registry_name: str, callback: Callable[[], None]
    ) -> None:
        """Register a callback to be called when a registry is first accessed."""
        self._import_callbacks[registry_name] = callback

    def _ensure_registry_exists(self, registry_name: str) -> None:
        """Ensure a registry exists and trigger import callback if needed."""
        if registry_name not in self._registries:
            self._registries[registry_name] = {}
            if registry_name in self._import_callbacks:
                try:
                    self._import_callbacks[registry_name]()
                except Exception as e:
                    self._logger.warning(
                        f"Failed to execute import callback for {registry_name}: {e}"
                    )

    def register(self, registry_name: str, name: str, component: Any) -> None:
        """Register a component in the specified registry."""
        self._ensure_registry_exists(registry_name)

        if name in self._registries[registry_name]:
            self._logger.warning(
                f"Component '{name}' already registered in '{registry_name}'. Overwriting."
            )

        self._registries[registry_name][name] = component
        self._logger.debug(f"Registered '{name}' in '{registry_name}'")

    def get(self, registry_name: str, name: str, default: Optional[Any] = None) -> Any:
        """Get a component from the specified registry."""
        self._ensure_registry_exists(registry_name)

        if name not in self._registries[registry_name]:
            if default is not None:
                return default
            available = list(self._registries[registry_name].keys())
            raise ValueError(
                f"Component '{name}' not found in '{registry_name}'. Available: {available}"
            )

        return self._registries[registry_name][name]

    def list_components(self, registry_name: str) -> List[str]:
        """List all components in a registry."""
        self._ensure_registry_exists(registry_name)
        return list(self._registries[registry_name].keys())

    def clear(self, registry_name: Optional[str] = None) -> None:
        """Clear a specific registry or all registries."""
        if registry_name is None:
            self._registries.clear()
        else:
            self._registries[registry_name] = {}

    @contextmanager
    def temporary_registry(self, registry_name: str) -> Iterator[None]:
        """Context manager for temporary registry operations."""
        original = self._registries.get(registry_name, {}).copy()
        try:
            yield
        finally:
            self._registries[registry_name] = original


# Global registry instance
_registry = SafeRegistry()


def get_registry() -> SafeRegistry:
    """Get the global registry instance."""
    return _registry


def register_component(registry_name: str, name: str) -> Callable[[T], T]:
    """Decorator to register a component in a specific registry."""

    def decorator(component: T) -> T:
        _registry.register(registry_name, name, component)
        return component

    return decorator


def get_component(registry_name: str, name: str, default: Optional[Any] = None) -> Any:
    """Get a component from a specific registry."""
    return _registry.get(registry_name, name, default)


def list_components(registry_name: str) -> List[str]:
    """List all components in a registry."""
    return _registry.list_components(registry_name)


def clear_registry(registry_name: Optional[str] = None) -> None:
    """Clear a specific registry or all registries."""
    _registry.clear(registry_name)


# Drop-in replacement functions for existing registry
def register_dataset(name: str) -> Callable[[T], T]:
    """Register a dataset in the dataset registry."""
    return register_component("datasets", name)


def get_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
    """Get a dataset from the dataset registry with optional arguments."""
    dataset_cls = get_component("datasets", name)
    if dataset_cls is None:
        # Try to find in torchvision datasets as fallback
        try:
            from torchvision import datasets  # type: ignore

            if hasattr(datasets, name):
                return getattr(datasets, name)(*args, **kwargs)
        except ImportError:
            pass

        available_datasets = list_components("datasets")
        raise ValueError(f"Dataset '{name}' not found. Available: {available_datasets}")

    return dataset_cls(*args, **kwargs)


def register_model(name: str) -> Callable[[T], T]:
    """Register a model in the model registry."""
    return register_component("models", name)


def get_model(name: str) -> Any:
    """Get a model from the model registry."""
    return get_component("models", name)


def register_loss(name: str) -> Callable[[T], T]:
    """Register a loss function in the loss registry."""
    return register_component("losses", name)


def get_loss(name: str) -> Any:
    """Get a loss function from the loss registry."""
    return get_component("losses", name)


def register_trainer(name: str) -> Callable[[T], T]:
    """Register a trainer in the trainer registry."""
    return register_component("trainers", name)


def get_trainer(name: str) -> Any:
    """Get a trainer from the trainer registry."""
    return get_component("trainers", name)


def register_transform(name: str) -> Callable[[T], T]:
    """Register a transform in the transform registry."""
    return register_component("transforms", name)


def get_transform(name: str) -> Any:
    """Get a transform from the transform registry."""
    return get_component("transforms", name)


# Import callbacks for automatic registration
def _register_import_callbacks() -> None:
    """Register import callbacks for automatic component registration."""

    def import_models() -> None:
        """Import all model modules to trigger registration."""
        import refrakt_core.models  # noqa

    def import_datasets() -> None:
        """Import all dataset modules to trigger registration."""
        import refrakt_core.datasets  # noqa

    def import_losses() -> None:
        """Import all loss modules to trigger registration."""
        import refrakt_core.losses  # noqa

    def import_trainers() -> None:
        """Import all trainer modules to trigger registration."""
        import refrakt_core.trainer  # noqa

    def import_transforms() -> None:
        """Import all transform modules to trigger registration."""
        import refrakt_core.transforms  # noqa

    _registry.register_import_callback("models", import_models)
    _registry.register_import_callback("datasets", import_datasets)
    _registry.register_import_callback("losses", import_losses)
    _registry.register_import_callback("trainers", import_trainers)
    _registry.register_import_callback("transforms", import_transforms)


# Initialize import callbacks
_register_import_callbacks()
