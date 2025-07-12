"""
Model utilities for Refrakt.

This module provides utility functions for model building and wrapping operations,
extracted from the main model_builder to reduce complexity and improve maintainability.

The module handles:
- Model configuration validation and parsing
- Configuration override application
- Base model instantiation from registry
- Model wrapping with specialized wrappers
- Default wrapper creation as fallback
- Fusion block integration for ensemble models

These utilities are designed to work with the model builder pipeline and provide
robust error handling and fallback mechanisms for model construction.

Typical usage involves passing configuration dictionaries, module registries, and
device specifications to build and wrap models for training or inference.
"""

import inspect
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from refrakt_core.hooks.hyperparameter_override import apply_overrides
from refrakt_core.wrappers.schema.default_model import DefaultModelWrapper


def validate_model_config(cfg_dict: Any) -> tuple[str, Dict[str, Any], Optional[str]]:
    """
    Validate and extract model configuration parameters.

    This function parses the model configuration dictionary to extract the model name,
    parameters, and optional wrapper specification. It performs type validation to
    ensure the configuration structure is correct before model building.

    Args:
        cfg_dict: Configuration dictionary containing model specifications.
                  Expected to have a 'model' key with 'name', 'params', and \
                  optional 'wrapper' fields.

    Returns:
        Tuple containing:
        - model_name (str): The name of the model to instantiate
        - model_params (Dict[str, Any]): Model-specific parameters
        - wrapper_name (Optional[str]): Optional wrapper name for model wrapping

    Raises:
        TypeError: If the configuration structure is invalid or required fields
                  are missing or of incorrect type.
    """
    model_cfg = cfg_dict.get("model")
    if not isinstance(model_cfg, dict):
        raise TypeError(f"cfg.model must be a dict, got {type(model_cfg)}")

    model_params = model_cfg.get("params", {}) or {}
    model_name = model_cfg.get("name")
    if not isinstance(model_name, str):
        raise TypeError(f"model_name must be a str, got {type(model_name)}")

    wrapper_name = model_cfg.get("wrapper", None)
    if wrapper_name is not None and not isinstance(wrapper_name, str):
        raise TypeError(f"wrapper_name must be a str or None, got {type(wrapper_name)}")

    return model_name, model_params, wrapper_name


def apply_model_overrides(cfg: Any, overrides: Optional[List[str]] = None) -> Any:
    """
    Apply configuration overrides to the model configuration.

    This function allows dynamic modification of model configuration parameters
    through override strings. Overrides are applied before model instantiation,
    enabling runtime parameter adjustments without modifying configuration files.

    Args:
        cfg: Configuration object (OmegaConf or DictConfig) to be modified
        overrides: Optional list of override strings in format 'path.to.param=value'
                  to modify configuration before model building

    Returns:
        Updated configuration object with overrides applied

    Note:
        Overrides are applied using the hyperparameter override system,
        which supports nested parameter modification.
    """
    if overrides:
        # Convert to dict, apply overrides, then back to OmegaConf
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg_dict, dict):
            cfg_dict = apply_overrides(OmegaConf.create(cfg_dict), overrides)
            cfg = OmegaConf.create(cfg_dict)

    return cfg


def instantiate_base_model(
    model_name: str, model_params: Dict[str, Any], modules: Dict[str, Any], device: str
) -> Any:
    """
    Instantiate the base model from the registry.

    This function retrieves the model factory function from the modules registry
    and instantiates the base model with the provided parameters. It handles
    parameter conversion and special cases like AutoEncoder type/mode mapping.

    Args:
        model_name: Name of the model to instantiate from the registry
        model_params: Model-specific parameters dictionary
        modules: Registry dictionary containing available model factory functions
        device: Target device string (e.g., "cuda", "cpu") for model placement

    Returns:
        Instantiated model object moved to the specified device

    Raises:
        ValueError: If the get_model function is not found in the modules registry
        Exception: Any exceptions during model instantiation are propagated
    """
    get_model_fn = modules.get("get_model")
    if get_model_fn is None:
        raise ValueError("[ERROR] get_model function not found in modules registry.")

    # Convert DictConfig to regular dict if needed
    model_params_dict = (
        dict(model_params) if hasattr(model_params, "items") else model_params
    )

    # Patch for AutoEncoder: map 'type' to 'mode' if present
    if model_name == "autoencoder" and "type" in model_params_dict:
        model_params_dict["mode"] = model_params_dict.pop("type")

    raw_model = get_model_fn(model_name, **model_params_dict).to(device)
    return raw_model


def wrap_model(
    raw_model: Any,
    wrapper_name: str,
    model_params: Dict[str, Any],
    modules: Dict[str, Any],
    device: str,
) -> Any:
    """
    Wrap the model with the specified wrapper class.

    This function applies a wrapper to the base model using the wrapper registry.
    It automatically filters wrapper parameters based on the wrapper's constructor
    signature and handles special cases like AutoEncoder variant mapping.

    Args:
        raw_model: The base model object to be wrapped
        wrapper_name: Name of the wrapper class to apply from the registry
        model_params: Model parameters that may include wrapper-specific parameters
        modules: Registry dictionary containing available wrapper functions
        device: Target device string for the wrapped model

    Returns:
        Wrapped model object moved to the specified device

    Raises:
        ValueError: If the get_wrapper function is not found in the modules registry
                  or if the specified wrapper class is not found
    """
    get_wrapper_fn = modules.get("get_wrapper")
    if get_wrapper_fn is None:
        raise ValueError("[ERROR] get_wrapper function not found in modules registry.")

    wrapper_cls = get_wrapper_fn(wrapper_name)
    if wrapper_cls is None:
        raise ValueError(f"[ERROR] Wrapper class for '{wrapper_name}' not found.")

    sig = inspect.signature(wrapper_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {
        "self",
        "model",
        "args",
        "kwargs",
    }

    wrapper_args = {k: v for k, v in model_params.items() if k in valid_params}

    # Special handling for autoencoder wrapper: set 'variant' from \
    # model_params['mode'] if present
    if wrapper_name == "autoencoder" and "mode" in model_params:
        wrapper_args["variant"] = model_params["mode"]

    # Special handling for MSN wrapper: pass model_params instead of raw_model
    if wrapper_name == "msn":
        model = wrapper_cls(model=model_params, **wrapper_args).to(device)
    else:
        model = wrapper_cls(model=raw_model, **wrapper_args).to(device)
    
    print(f"[SUCCESS] Wrapped model with '{wrapper_name}'")
    return model


def create_default_wrapper(
    model_name: str, model_params: Dict[str, Any], modules: Dict[str, Any], device: str
) -> Any:
    """
    Create a default model wrapper as a fallback mechanism.

    This function is used when no specific wrapper is specified in the configuration.
    It creates a DefaultModelWrapper that provides a standardized interface for
    models that don't require specialized wrapping.

    Args:
        model_name: Name of the model for the default wrapper
        model_params: Model parameters to be stored in the wrapper
        modules: Registry dictionary containing available functions
        device: Target device string for the wrapped model

    Returns:
        DefaultModelWrapper instance moved to the specified device

    Note:
        This function provides a safety net for models that don't specify
        a custom wrapper, ensuring all models have a consistent interface.
    """
    print(
        f"[INFO] No wrapper specified. Using DefaultModelWrapper for model \
            '{model_name}'"
    )
    model = DefaultModelWrapper(
        model_name=model_name, model_params=model_params, modules=modules
    ).to(device)
    return model


def add_fusion_block(model: Any, model_cfg: Any, device: str) -> Any:
    """
    Add a fusion block to the model if specified in configuration.

    This function optionally wraps the model with a FusionBlock for ensemble
    or multi-modal architectures. The fusion block is only added if fusion
    configuration is present in the model configuration.

    Args:
        model: The model object to potentially wrap with fusion block
        model_cfg: Model configuration that may contain fusion settings
        device: Target device string for the fusion block

    Returns:
        Either the original model or the model wrapped with FusionBlock,
        depending on whether fusion configuration is present

    Note:
        Fusion blocks are used for advanced architectures that combine
        multiple models or modalities into a single ensemble model.
    """
    fusion_cfg = model_cfg.get("fusion", None)
    if fusion_cfg:
        from refrakt_core.integrations.fusion.block import FusionBlock

        print(
            f"[INFO] Wrapping model with FusionBlock using fusion config: {fusion_cfg}"
        )
        model = FusionBlock(backbone=model, fusion_cfg=fusion_cfg).to(device)
        print("[SUCCESS] Model wrapped with FusionBlock.")

    return model
