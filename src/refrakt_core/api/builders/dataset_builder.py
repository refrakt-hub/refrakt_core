"""
Dataset builder for Refrakt.

This module provides utilities to construct datasets and dataset wrappers from
configuration dictionaries. It supports both standard and wrapped datasets, and
integrates with the transform builder for preprocessing pipelines.

The module handles:
- Dataset configuration validation and parsing
- Base dataset instantiation from registry
- Dataset wrapper application for specialized use cases
- Transform pipeline integration and application
- Parameter type checking and validation
- Registry-based dataset discovery and instantiation
- Support for contrastive and other wrapped datasets

Typical usage involves passing a configuration (OmegaConf DictConfig) describing
the dataset, wrapper, and transforms.
"""

from typing import Any

from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.builders.transform_builder import build_transform
from refrakt_core.registry.dataset_registry import DATASET_REGISTRY, get_dataset


def build_dataset(cfg: DictConfig) -> Any:
    """
    Build a dataset or wrapped dataset from configuration.

    This function supports both standard datasets and wrapped datasets (e.g., for
    contrastive learning). It integrates with the transform builder to apply
    preprocessing pipelines and handles parameter validation and type checking.

    The function follows a multi-step process:
    1. Validate and parse configuration parameters
    2. Extract dataset name, parameters, and optional wrapper
    3. Build transform pipeline if specified
    4. Create base dataset or wrapped dataset
    5. Apply transforms and return final dataset

    Args:
        cfg: Configuration object (DictConfig) specifying dataset parameters.
            Expected keys include 'name', 'params', 'wrapper', and 'transform'.
            If 'wrapper' is specified, the base dataset is wrapped accordingly.

    Returns:
        The instantiated dataset or wrapped dataset object. The exact type depends
        on the dataset and wrapper configuration.

    Raises:
        TypeError: If the configuration or its fields are not of the expected type
        ValueError: If the specified dataset or wrapper is not found in the registry
    """
    # Convert to native Python types for compatibility
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Ensure cfg_dict is a dict
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"cfg_dict must be a dict, got {type(cfg_dict)}")

    # Extract dataset parameters
    params = cfg_dict.get("params", {})
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dict, got {type(params)}")
    dataset_params = params.copy()

    dataset_name = cfg_dict.get("name")
    if not isinstance(dataset_name, str):
        raise TypeError(f"dataset_name must be a str, got {type(dataset_name)}")

    wrapper_name = cfg_dict.get("wrapper", None)
    if wrapper_name is not None and not isinstance(wrapper_name, str):
        raise TypeError(f"wrapper_name must be a str or None, got {type(wrapper_name)}")

    # Handle transform separately
    transform_cfg = cfg_dict.get("transform", None)
    transform_fn = build_transform(transform_cfg) if transform_cfg else None

    # Handle wrapped datasets (e.g., contrastive)
    if wrapper_name:
        # Create base dataset without transform
        base_dataset = get_dataset(dataset_name, **dataset_params)

        # Get wrapper class from registry
        if wrapper_name not in DATASET_REGISTRY:
            raise ValueError(f"Wrapper dataset '{wrapper_name}' not found in registry")

        wrapper_cls = DATASET_REGISTRY[wrapper_name]

        # Pass base_dataset to wrapper
        return wrapper_cls(base_dataset, transform=transform_fn)
    # For non-wrapped datasets, apply transform directly
    if transform_fn:
        dataset_params["transform"] = transform_fn
    return get_dataset(dataset_name, **dataset_params)
