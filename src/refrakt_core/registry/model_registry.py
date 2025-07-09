"""Model registry for managing model classes."""

from typing import Any, Callable, Dict, Type

from refrakt_core.global_logging import get_global_logger

MODEL_REGISTRY: Dict[str, Type[Any]] = {}
_IMPORTED: bool = False


def register_model(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Register a model class with the given name.

    Args:
        name: The name to register the model under.

    Returns:
        A decorator function that registers the model class.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        logger = get_global_logger()
        if name in MODEL_REGISTRY:
            logger.debug("Warning: Model '%s' already registered. Skipping.", name)
            return cls
        logger.debug("Registering model: %s", name)
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def _import_models() -> None:
    """Import all model modules to trigger registration."""
    global _IMPORTED  # pylint: disable=global-statement
    if not _IMPORTED:
        try:
            import refrakt_core.models

            _IMPORTED = True
        except ImportError as e:
            logger = get_global_logger()
            logger.error(f"Failed to import models: {e}")


def get_model(name: str, *args: Any, **kwargs: Any) -> Any:
    """Get model instance by name with optional arguments.

    Args:
        name: The name of the model to retrieve.
        *args: Positional arguments to pass to the model constructor.
        **kwargs: Keyword arguments to pass to the model constructor.

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If the model is not found.
    """
    _import_models()
    if name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available: {available_models}")
    return MODEL_REGISTRY[name](*args, **kwargs)


def log_registry_id() -> None:
    """Log the registry ID for debugging purposes."""
    logger = get_global_logger()
    logger.debug("MODEL REGISTRY ID: %s", id(MODEL_REGISTRY))
