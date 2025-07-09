"""Trainer registry for managing trainer classes."""

from typing import Any, Callable, Dict, Type

from refrakt_core.global_logging import get_global_logger

TRAINER_REGISTRY: Dict[str, Type[Any]] = {}
_IMPORTED: bool = False


def register_trainer(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Register a trainer class with the given name.

    Args:
        name: The name to register the trainer under.

    Returns:
        A decorator function that registers the trainer class.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        logger = get_global_logger()
        logger.debug("Registering trainer: %s", name)
        TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


def _import_trainers() -> None:
    """Import all trainer modules to trigger registration."""
    global _IMPORTED  # pylint: disable=global-statement
    if not _IMPORTED:
        try:
            import refrakt_core.trainer

            _IMPORTED = True
        except ImportError as e:
            logger = get_global_logger()
            logger.error(f"Failed to import trainers: {e}")


def get_trainer(name: str) -> Type[Any]:
    """Get trainer class by name.

    Args:
        name: The name of the trainer to retrieve.

    Returns:
        The trainer class (not an instance).

    Raises:
        ValueError: If the trainer is not found.
    """
    _import_trainers()
    if name not in TRAINER_REGISTRY:
        available_trainers = list(TRAINER_REGISTRY.keys())
        raise ValueError(f"Trainer '{name}' not found. Available: {available_trainers}")
    return TRAINER_REGISTRY[name]  # Return the class, not an instance


def log_registry_id() -> None:
    """Log the registry ID for debugging purposes."""
    logger = get_global_logger()
    logger.debug("TRAINER REGISTRY ID: %s", id(TRAINER_REGISTRY))
