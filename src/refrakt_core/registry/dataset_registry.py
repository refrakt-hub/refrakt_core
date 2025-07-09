"""Dataset registry for managing dataset classes."""

from typing import Any, Callable, Dict, Type

from refrakt_core.global_logging import get_global_logger

DATASET_REGISTRY: Dict[str, Type[Any]] = {}
_IMPORTED: bool = False


def register_dataset(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Register a dataset class with the given name.

    Args:
        name: The name to register the dataset under.

    Returns:
        A decorator function that registers the dataset class.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        logger = get_global_logger()
        if name in DATASET_REGISTRY:
            logger.debug("Warning: Dataset '%s' already registered. Skipping.", name)
            return cls
        logger.debug("Registering dataset: %s", name)
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def _import_datasets() -> None:
    """Import all dataset modules to trigger registration."""
    global _IMPORTED  # pylint: disable=global-statement
    if not _IMPORTED:
        try:
            import refrakt_core.datasets

            _IMPORTED = True
        except ImportError as e:
            logger = get_global_logger()
            logger.error(f"Failed to import datasets: {e}")


def get_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
    """Get dataset instance by name with optional arguments.

    Args:
        name: The name of the dataset to retrieve.
        *args: Positional arguments to pass to the dataset constructor.
        **kwargs: Keyword arguments to pass to the dataset constructor.

    Returns:
        An instance of the requested dataset.

    Raises:
        ValueError: If the dataset is not found.
    """
    _import_datasets()

    if name not in DATASET_REGISTRY:
        # Try to find in torchvision datasets as fallback
        try:
            from torchvision import (
                datasets,
            )  # type: ignore  # pylint: disable=import-outside-toplevel

            if hasattr(datasets, name):
                return getattr(datasets, name)(*args, **kwargs)
        except ImportError:
            pass

        available_datasets = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Dataset '{name}' not found. Available: {available_datasets}")

    return DATASET_REGISTRY[name](*args, **kwargs)


def log_registry_id() -> None:
    """Log the registry ID for debugging purposes."""
    logger = get_global_logger()
    logger.debug("DATASET REGISTRY ID: %s", id(DATASET_REGISTRY))
