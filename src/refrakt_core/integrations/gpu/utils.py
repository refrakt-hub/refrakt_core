"""
Utility functions for GPU integration wrapper decomposition.
"""

import importlib
from typing import Any, Dict, Type

from refrakt_core.integrations.gpu.registry import load_cuml_registry


def extract_wrapper_params(
    params: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract wrapper-specific parameters from the input parameters.

    Args:
        params: Input parameters dictionary

    Returns:
        Tuple of (wrapper_params, model_params)
    """
    wrapper_params: Dict[str, Any] = {}
    model_params = dict(params)  # Make a copy to modify

    # Handle special parameters
    if "fusion_head" in model_params:
        fusion_head_val = model_params.pop("fusion_head")
        if fusion_head_val is None:
            wrapper_params["fusion_head"] = {}
        else:
            wrapper_params["fusion_head"] = fusion_head_val

    return wrapper_params, model_params


def instantiate_cuml_model(model: str, model_params: Dict[str, Any]) -> Any:
    """
    Instantiate a cuML model from registry or full class path.

    Args:
        model: Model key or full class path
        model_params: Parameters for model instantiation

    Returns:
        Instantiated cuML model

    Raises:
        ValueError: If the model path is invalid
    """
    registry = load_cuml_registry()
    class_path = registry.get(model, model)

    module_path, class_name = class_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
        model_cls: Type[object] = getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Invalid cuML model '{model}': {e}")

    return model_cls(**model_params)


def validate_predict_proba_support(model: Any) -> None:
    """
    Validate that the model supports predict_proba method.

    Args:
        model: The model to validate

    Raises:
        AttributeError: If model doesn't support predict_proba
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{model.__class__.__name__} does not support predict_proba"
        )
