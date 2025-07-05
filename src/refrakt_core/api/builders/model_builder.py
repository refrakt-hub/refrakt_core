"""
Model builder for Refrakt.

This module provides utilities to construct and wrap models from configuration dictionaries.
It supports model instantiation, optional wrapping, and fusion block integration for advanced architectures.

Typical usage involves passing a configuration (OmegaConf), a modules registry, and a device to build and wrap models for training or inference.
"""

from typing import Any, Dict, List, Optional, Union

from omegaconf import OmegaConf, DictConfig

from .utils.model_utils import (
    validate_model_config,
    apply_model_overrides,
    instantiate_base_model,
    wrap_model,
    create_default_wrapper,
    add_fusion_block,
)


def build_model(
    cfg: Union[OmegaConf, DictConfig], 
    modules: Dict[str, Any], 
    device: str,
    overrides: Optional[List[str]] = None
) -> Any:
    """
    Build and wrap a model from configuration, with optional fusion block integration.

    This function instantiates a model using the provided configuration and modules registry.
    It supports optional model wrapping (e.g., for autoencoders) and can further wrap the model with a FusionBlock if specified.
    Robust error handling ensures a DefaultModelWrapper is used as a fallback.

    Args:
        cfg (OmegaConf): Configuration specifying the model structure and parameters.
        modules (Dict[str, Any]): Registry of available model and wrapper functions.
        device (str): Device on which to place the model.
        overrides (Optional[List[str]]): List of override strings in format 'path.to.param=value'

    Returns:
        Any: Instantiated and wrapped model object, ready for training or inference.

    Raises:
        TypeError: If the configuration or its fields are not of the expected type.
        ValueError: If required model or wrapper components are missing or not found in the registry.
    """
    import refrakt_core.models

    # Apply overrides if provided
    cfg = apply_model_overrides(cfg, overrides)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg must convert to a dict, got {type(cfg_dict)}")
    
    model_name, model_params, wrapper_name = validate_model_config(cfg_dict)

    try:
        # Step 1: Instantiate base model
        raw_model = instantiate_base_model(model_name, model_params, modules, device)

        # Step 2: Wrap model (if wrapper is specified)
        if wrapper_name:
            model = wrap_model(raw_model, wrapper_name, model_params, modules, device)
        else:
            model = create_default_wrapper(model_name, model_params, modules, device)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FALLBACK] Using DefaultModelWrapper due to error: {e}")
        model = create_default_wrapper(model_name, model_params, modules, device)

    # Step 3: Add fusion block if specified
    model_cfg = cfg_dict.get("model")
    model = add_fusion_block(model, model_cfg, device)

    print(f"[FINALIZED] Model: {model_name} with params: {model_params}")
    return model
