"""Transform registry for managing transform classes."""

from typing import Any, Callable, Dict, Type

from refrakt_core.logging import get_global_logger

TRANSFORM_REGISTRY: Dict[str, Type[Any]] = {}
_IMPORTED: bool = False


def register_transform(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Register a transform class with the given name.
    
    Args:
        name: The name to register the transform under.
        
    Returns:
        A decorator function that registers the transform class.
    """
    def decorator(cls: Type[Any]) -> Type[Any]:
        logger = get_global_logger()
        if name in TRANSFORM_REGISTRY:
            logger.debug("Warning: Transform '%s' already registered. Skipping.", name)
            return cls
        logger.debug("Registering transform: %s", name)
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return decorator


def get_transform(name: str, *args: Any, **kwargs: Any) -> Any:
    """Get transform instance by name with optional arguments.
    
    Args:
        name: The name of the transform to retrieve.
        *args: Positional arguments to pass to the transform constructor.
        **kwargs: Keyword arguments to pass to the transform constructor.
        
    Returns:
        An instance of the requested transform.
        
    Raises:
        ValueError: If the transform is not found.
    """
    global _IMPORTED  # pylint: disable=global-statement
    if not _IMPORTED:
        # Trigger import of transforms module to register custom transforms
        _IMPORTED = True

    if name not in TRANSFORM_REGISTRY:
        # Try to find in torchvision transforms as fallback
        try:
            from torchvision import \
                transforms  # pylint: disable=import-outside-toplevel

            if hasattr(transforms, name):
                return getattr(transforms, name)(*args, **kwargs)
        except ImportError:
            pass

        available_transforms = list(TRANSFORM_REGISTRY.keys()) + [
            "ToTensor",
            "Normalize", 
            "Compose",
        ]  # Example torchvision names
        raise ValueError(f"Transform '{name}' not found. Available: {available_transforms}")

    return TRANSFORM_REGISTRY[name](*args, **kwargs)


def log_registry_id() -> None:
    """Log the registry ID for debugging purposes."""
    logger = get_global_logger()
    logger.debug("TRANSFORM REGISTRY ID: %s", id(TRANSFORM_REGISTRY))
