"""Loss registry for managing loss functions and classes."""

from typing import Any, Callable, Dict, Type, Union

from refrakt_core.logging import get_global_logger

LOSS_REGISTRY: Dict[str, Union[Type[Any], Callable[..., Any]]] = {}
_IMPORTED: bool = False


def register_loss(name: str) -> Callable[[Union[Type[Any],
                                        Callable[..., Any]]],
                                        Union[Type[Any],
                                        Callable[..., Any]]]:
    """Register a loss class or function with the given name.
    
    Args:
        name: The name to register the loss under.
        
    Returns:
        A decorator function that registers the loss 
        class or function.
    """
    def decorator(cls_or_fn: Union[Type[Any], Callable[..., Any]]) -> Union[Type[Any], Callable[..., Any]]:
        logger = get_global_logger()
        if name in LOSS_REGISTRY:
            logger.debug("Warning: Loss '%s' already registered. Skipping.", name)
            return cls_or_fn
        logger.debug("Registering loss: %s", name)
        LOSS_REGISTRY[name] = cls_or_fn
        return cls_or_fn

    return decorator


def get_loss(name: str, *args: Any, **kwargs: Any) -> Any:
    """Get loss instance by name with optional arguments.
    
    Args:
        name: The name of the loss to retrieve.
        *args: Positional arguments to pass to the loss constructor.
        **kwargs: Keyword arguments to pass to the loss constructor.
        
    Returns:
        An instance of the requested loss.
        
    Raises:
        ValueError: If the loss is not found.
    """
    global _IMPORTED  # pylint: disable=global-statement
    if not _IMPORTED:
        # Auto-import custom losses
        _IMPORTED = True

        # Add standard PyTorch losses to registry
        from torch import nn  # pylint: disable=import-outside-toplevel

        standard_losses = {
            "mse": nn.MSELoss,
            "l1": nn.L1Loss,
            "bce": nn.BCELoss,
        }

        for loss_name, loss_class in standard_losses.items():
            if loss_name not in LOSS_REGISTRY:
                register_loss(loss_name)(loss_class)

    if name not in LOSS_REGISTRY:
        available_losses = list(LOSS_REGISTRY.keys())
        raise ValueError(
            f"Loss '{name}' not found. Available: {available_losses}"
        )

    return LOSS_REGISTRY[name](*args, **kwargs)


def log_registry_id() -> None:
    """Log the registry ID for debugging purposes."""
    logger = get_global_logger()
    logger.debug("LOSS REGISTRY ID: %s", id(LOSS_REGISTRY))
