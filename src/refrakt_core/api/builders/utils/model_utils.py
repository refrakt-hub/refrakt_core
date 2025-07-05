"""
Model utilities for Refrakt.

This module provides utility functions for model building and wrapping,
extracted from the main model_builder to reduce complexity.
"""

import inspect
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf, DictConfig
from refrakt_core.wrappers.schema.default_model import DefaultModelWrapper
from refrakt_core.hooks.hyperparameter_override import apply_overrides


def validate_model_config(cfg_dict: Any) -> tuple[str, Dict[str, Any], Optional[str]]:
    """
    Validate and extract model configuration.
    
    Args:
        cfg_dict: Configuration dictionary
        
    Returns:
        Tuple of (model_name, model_params, wrapper_name)
        
    Raises:
        TypeError: If configuration is invalid
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
    Apply overrides to model configuration.
    
    Args:
        cfg: Configuration object
        overrides: List of override strings
        
    Returns:
        Updated configuration
    """
    if overrides:
        # Convert to dict, apply overrides, then back to OmegaConf
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg_dict, dict):
            cfg_dict = apply_overrides(OmegaConf.create(cfg_dict), overrides)
            cfg = OmegaConf.create(cfg_dict)
    
    return cfg


def instantiate_base_model(
    model_name: str,
    model_params: Dict[str, Any],
    modules: Dict[str, Any],
    device: str
) -> Any:
    """
    Instantiate the base model.
    
    Args:
        model_name: Name of the model
        model_params: Model parameters
        modules: Registry of available functions
        device: Device to place model on
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If required components are missing
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
    device: str
) -> Any:
    """
    Wrap the model with the specified wrapper.
    
    Args:
        raw_model: The base model
        wrapper_name: Name of the wrapper to use
        model_params: Model parameters
        modules: Registry of available functions
        device: Device to place model on
        
    Returns:
        Wrapped model
        
    Raises:
        ValueError: If wrapper is not found
    """
    get_wrapper_fn = modules.get("get_wrapper")
    if get_wrapper_fn is None:
        raise ValueError("[ERROR] get_wrapper function not found in modules registry.")
    
    wrapper_cls = get_wrapper_fn(wrapper_name)
    if wrapper_cls is None:
        raise ValueError(f"[ERROR] Wrapper class for '{wrapper_name}' not found.")
    
    sig = inspect.signature(wrapper_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {
        "self", "model", "args", "kwargs",
    }

    wrapper_args = {k: v for k, v in model_params.items() if k in valid_params}

    # Special handling for autoencoder wrapper: set 'variant' from model_params['mode'] if present
    if wrapper_name == "autoencoder" and "mode" in model_params:
        wrapper_args["variant"] = model_params["mode"]

    model = wrapper_cls(model=raw_model, **wrapper_args).to(device)
    print(f"[SUCCESS] Wrapped model with '{wrapper_name}'")
    return model


def create_default_wrapper(
    model_name: str,
    model_params: Dict[str, Any],
    modules: Dict[str, Any],
    device: str
) -> Any:
    """
    Create a default model wrapper.
    
    Args:
        model_name: Name of the model
        model_params: Model parameters
        modules: Registry of available functions
        device: Device to place model on
        
    Returns:
        Default wrapped model
    """
    print(f"[INFO] No wrapper specified. Using DefaultModelWrapper for model '{model_name}'")
    model = DefaultModelWrapper(
        model_name=model_name, model_params=model_params, modules=modules
    ).to(device)
    return model


def add_fusion_block(
    model: Any,
    model_cfg: Any,
    device: str
) -> Any:
    """
    Add fusion block to the model if specified.
    
    Args:
        model: The model to wrap
        model_cfg: Model configuration
        device: Device to place model on
        
    Returns:
        Model with fusion block if specified
    """
    fusion_cfg = model_cfg.get("fusion", None)
    if fusion_cfg:
        from refrakt_core.integrations.fusion.block import FusionBlock

        print(f"[INFO] Wrapping model with FusionBlock using fusion config: {fusion_cfg}")
        model = FusionBlock(backbone=model, fusion_cfg=fusion_cfg).to(device)
        print("[SUCCESS] Model wrapped with FusionBlock.")
    
    return model 