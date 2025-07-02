"""
Model builder for Refrakt.

This module provides utilities to construct and wrap models from configuration dictionaries.
It supports model instantiation, optional wrapping, and fusion block integration for advanced architectures.

Typical usage involves passing a configuration (OmegaConf), a modules registry, and a device to build and wrap models for training or inference.
"""

import inspect
from typing import Any, Dict

from omegaconf import OmegaConf
from refrakt_core.integrations.fusion.block import FusionBlock
from refrakt_core.registry.wrapper_registry import load_wrapper
from refrakt_core.wrappers.schema.default_model import DefaultModelWrapper


def build_model(cfg: OmegaConf, modules: Dict[str, Any], device: str) -> Any:
    """
    Build and wrap a model from configuration, with optional fusion block integration.

    This function instantiates a model using the provided configuration and modules registry.
    It supports optional model wrapping (e.g., for autoencoders) and can further wrap the model with a FusionBlock if specified.
    Robust error handling ensures a DefaultModelWrapper is used as a fallback.

    Args:
        cfg (OmegaConf): Configuration specifying the model structure and parameters.
        modules (Dict[str, Any]): Registry of available model and wrapper functions.
        device (str): Device on which to place the model.

    Returns:
        Any: Instantiated and wrapped model object, ready for training or inference.

    Raises:
        TypeError: If the configuration or its fields are not of the expected type.
        ValueError: If required model or wrapper components are missing or not found in the registry.
    """
    import refrakt_core.models

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg must convert to a dict, got {type(cfg_dict)}")
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

    try:
        get_model_fn = modules.get("get_model")
        if get_model_fn is None:
            raise ValueError(
                "[ERROR] get_model function not found in modules registry."
            )

        get_wrapper_fn = modules.get("get_wrapper")

        # Convert DictConfig to regular dict if needed
        model_params_dict = (
            dict(model_params) if hasattr(model_params, "items") else model_params
        )

        # Step 1: Instantiate base model
        model_cls = get_model_fn(model_name)
        # Patch for AutoEncoder: map 'type' to 'mode' if present
        if model_name == "autoencoder" and "type" in model_params_dict:
            model_params_dict["mode"] = model_params_dict.pop("type")
        raw_model = model_cls(**model_params_dict).to(device)

        # Step 2: Wrap model (if wrapper is specified)
        if wrapper_name:
            if get_wrapper_fn is None:
                raise ValueError(
                    "[ERROR] get_wrapper function not found in modules registry."
                )
            wrapper_cls = get_wrapper_fn(wrapper_name)
            if wrapper_cls is None:
                raise ValueError(
                    f"[ERROR] Wrapper class for '{wrapper_name}' not found."
                )
            sig = inspect.signature(wrapper_cls.__init__)
            valid_params = set(sig.parameters.keys()) - {
                "self",
                "model",
                "args",
                "kwargs",
            }

            wrapper_args = {k: v for k, v in model_params.items() if k in valid_params}

            # Special handling for autoencoder wrapper: set 'variant' from model_params['mode'] if present
            if wrapper_name == "autoencoder" and "mode" in model_params:
                wrapper_args["variant"] = model_params["mode"]

            model = wrapper_cls(model=raw_model, **wrapper_args).to(device)
            print(f"[SUCCESS] Wrapped model '{model_name}' with '{wrapper_name}'")
        else:
            print(
                f"[INFO] No wrapper specified. Using DefaultModelWrapper for model '{model_name}'"
            )
            model = DefaultModelWrapper(
                model_name=model_name, model_params=model_params, modules=modules
            ).to(device)

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[FALLBACK] Using DefaultModelWrapper due to error: {e}")
        model = DefaultModelWrapper(
            model_name=model_name, model_params=model_params, modules=modules
        ).to(device)

    fusion_cfg = model_cfg.get("fusion", None)
    if fusion_cfg:
        from refrakt_core.integrations.fusion.block import FusionBlock

        print(
            f"[INFO] Wrapping model with FusionBlock using fusion config: {fusion_cfg}"
        )
        model = FusionBlock(backbone=model, fusion_cfg=fusion_cfg).to(device)
        print("[SUCCESS] Model wrapped with FusionBlock.")

    print(f"[FINALIZED] Model: {model_name} with params: {model_params}")
    return model
