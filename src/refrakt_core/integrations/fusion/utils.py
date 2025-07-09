"""
Utility functions for fusion builder decomposition.
"""

from typing import Any, Dict, Union

from refrakt_core.integrations.cpu.wrapper import SklearnWrapper
from refrakt_core.integrations.gpu.wrapper import CuMLWrapper


def create_sklearn_wrapper(model: str, params: Dict[str, Any]) -> SklearnWrapper:
    """
    Create a sklearn wrapper with the given model and parameters.

    Args:
        model: Model key or class path
        params: Model parameters

    Returns:
        SklearnWrapper instance
    """
    return SklearnWrapper(model, **params)


def create_cuml_wrapper(model: str, params: Dict[str, Any]) -> CuMLWrapper:
    """
    Create a cuML wrapper with the given model and parameters.

    Args:
        model: Model key or class path
        params: Model parameters

    Returns:
        CuMLWrapper instance
    """
    return CuMLWrapper(model, **params)


def try_load_wrapper_from_path(
    wrapper: Union[SklearnWrapper, CuMLWrapper],
    model: str,
    fusion_head_config: Dict[str, Any],
) -> Union[SklearnWrapper, CuMLWrapper]:
    """
    Try to load a wrapper from a saved path.

    Args:
        wrapper: The wrapper instance to use as fallback
        model: Model key or class path
        fusion_head_config: Configuration containing the path

    Returns:
        Loaded wrapper or original wrapper if loading fails
    """
    if (
        fusion_head_config
        and isinstance(fusion_head_config, dict)
        and fusion_head_config.get("path")
    ):
        try:
            return wrapper.load(model, fusion_head_config["path"])
        except (FileNotFoundError, ValueError):
            return wrapper
    return wrapper


def validate_head_type(head_type: str) -> None:
    """
    Validate that the fusion head type is supported.

    Args:
        head_type: The head type to validate

    Raises:
        ValueError: If head type is unsupported
    """
    supported_types = ["sklearn", "cuml"]
    if head_type not in supported_types:
        raise ValueError(f"[FusionBuilder] Unsupported fusion head type: {head_type}")
