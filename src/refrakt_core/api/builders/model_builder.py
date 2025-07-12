"""
Model builder for Refrakt.

This module provides utilities to construct and wrap models from configuration
dictionaries. It supports model instantiation, optional wrapping, and fusion
block integration for advanced architectures.

The module handles:
- Model configuration validation and parsing
- Base model instantiation from registry
- Model wrapping with specialized wrappers
- Fusion block integration for ensemble models
- Fallback to default wrappers on errors
- Device placement and model optimization

Typical usage involves passing a configuration (OmegaConf), a modules registry,
and a device to build and wrap models for training or inference.
"""

from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf

from .utils.model_utils import (
    add_fusion_block,
    apply_model_overrides,
    create_default_wrapper,
    instantiate_base_model,
    validate_model_config,
    wrap_model,
)


def build_model(
    cfg: Union[OmegaConf, DictConfig],
    modules: Dict[str, Any],
    device: str,
    overrides: Optional[List[str]] = None,
) -> Any:
    """
    Build and wrap a model from configuration, with optional fusion block
    integration.

    This function instantiates a model using the provided configuration and
    modules registry. It supports optional model wrapping (e.g., for
    autoencoders) and can further wrap the model with a FusionBlock if
    specified. Robust error handling ensures a DefaultModelWrapper is used as a
    fallback.

    The function follows a multi-step process:
    1. Apply configuration overrides if provided
    2. Validate and parse model configuration
    3. Instantiate the base model from registry
    4. Apply optional model wrapper
    5. Add fusion block if specified
    6. Return the final model ready for training/inference

    Args:
        cfg: Configuration object (OmegaConf or DictConfig) specifying the model
            structure, parameters, and optional wrapper settings
        modules: Registry dictionary containing available model and wrapper
            functions
        device: Target device string (e.g., "cuda", "cpu") for model placement
        overrides: Optional list of override strings in format
            'path.to.param=value' to modify configuration before model building

    Returns:
        The instantiated and wrapped model object, ready for training or
        inference. The exact type depends on the model and wrapper
        configuration.

    Raises:
        TypeError: If the configuration or its fields are not of the expected
            type
        ValueError: If required model or wrapper components are missing or not
            found in the registry
        Exception: Any other exceptions during model building will trigger
            fallback to DefaultModelWrapper
    """

    # Apply overrides if provided
    cfg = apply_model_overrides(cfg, overrides)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg must convert to a dict, got {type(cfg_dict)}")

    model_name, model_params, wrapper_name = validate_model_config(cfg_dict)

    try:
        # Step 1: Instantiate base model (skip for special cases like MSN)
        if model_name == "msn" and wrapper_name == "msn":
            # MSN wrapper handles model instantiation
            raw_model = None
        else:
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
