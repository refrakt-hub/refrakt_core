"""Loss registry for managing loss functions and classes."""

from typing import Any, Callable, Dict, Optional, Type, Union

from refrakt_core.global_logging import get_global_logger

LOSS_REGISTRY: Dict[str, Union[Type[Any], Callable[..., Any]]] = {}
LOSS_MODES: Dict[str, str] = {}
_IMPORTED: bool = False


def register_loss(name: str, mode: Optional[str] = None) -> Callable[[Any], Any]:
    def decorator(cls_or_fn: Any) -> Any:
        logger = get_global_logger()
        if name in LOSS_REGISTRY:
            logger.debug("Warning: Loss '%s' already registered. Skipping.", name)
            return cls_or_fn

        logger.debug("Registering loss: %s", name)
        LOSS_REGISTRY[name] = cls_or_fn

        if mode:
            LOSS_MODES[name] = mode  # Register mode if given

        return cls_or_fn

    return decorator


def _import_losses() -> None:
    """Import all loss modules to trigger registration."""
    global _IMPORTED  # pylint: disable=global-statement
    if not _IMPORTED:
        try:
            # Import custom losses
            # Add standard PyTorch losses to registry
            from torch import nn  # pylint: disable=import-outside-toplevel

            import refrakt_core.losses

            standard_losses = {
                "mse": nn.MSELoss,
                "l1": nn.L1Loss,
                "bce": nn.BCELoss,
            }

            for loss_name, loss_class in standard_losses.items():
                if loss_name not in LOSS_REGISTRY:
                    register_loss(loss_name)(loss_class)

            _IMPORTED = True
        except ImportError as e:
            logger = get_global_logger()
            logger.error(f"Failed to import losses: {e}")


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
    _import_losses()
    if name not in LOSS_REGISTRY:
        available_losses = list(LOSS_REGISTRY.keys())
        raise ValueError(f"Loss '{name}' not found. Available: {available_losses}")

    return LOSS_REGISTRY[name](*args, **kwargs)


def get_loss_mode(name: str) -> str:
    return LOSS_MODES.get(name, "logits")


def log_registry_id() -> None:
    """Log the registry ID for debugging purposes."""
    logger = get_global_logger()
    logger.debug("LOSS REGISTRY ID: %s", id(LOSS_REGISTRY))
