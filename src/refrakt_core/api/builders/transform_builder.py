"""
Transform builder for Refrakt.

This module provides utilities to construct data transformation pipelines from configuration dictionaries.
It supports both simple and nested transforms, including custom and registry-based transforms, and integrates with PyTorch Compose.

Typical usage involves passing a configuration (list, dict, or ListConfig) describing the transform sequence or components.
"""

from typing import Any, Callable, Dict, List, Union

from omegaconf import ListConfig
from refrakt_core.registry.transform_registry import get_transform
from torchvision import transforms


def build_transform(cfg: Union[List[Any], ListConfig, Dict[str, Any]]) -> Callable:
    """
    Build a transform pipeline from configuration using the transform registry.

    This function supports both flat and nested transform configurations, including RandomApply and PairedTransform.
    It recursively builds and composes transforms as needed for flexible data preprocessing.

    Args:
        cfg (Union[List[Any], ListConfig, Dict[str, Any]]): Configuration specifying the transform sequence or components.
            Can be a list, ListConfig, or dict with 'views' or 'components' keys.

    Returns:
        Callable: A composed transform function or callable object.

    Raises:
        TypeError: If the configuration is not of a supported type.
        ValueError: If the configuration format is not recognized or supported.
    """
    transform_list = []

    # Resolve the actual transform sequence depending on style
    if isinstance(cfg, (list, ListConfig)):
        transform_sequence = cfg
    elif isinstance(cfg, dict):
        if "views" in cfg:
            transform_sequence = cfg["views"][0]
        elif "components" in cfg:
            transform_sequence = cfg["components"]
        else:
            raise ValueError(f"Unsupported transform configuration format: {cfg}")
    else:
        raise TypeError(f"Invalid transform config type: {type(cfg)}")

    transform_list = []
    for t in transform_sequence:
        name = t["name"]
        params = t.get("params", {})

        # Handle nested transform blocks like RandomApply
        if name.lower() == "randomapply":
            nested_cfgs = params.get("transforms", [])
            nested_transforms = build_transform(nested_cfgs)

            if isinstance(nested_transforms, transforms.Compose):
                nested_transforms = nested_transforms.transforms

            transform = get_transform(
                "RandomApply", nested_transforms, p=params.get("p", 0.5)
            )
        else:
            transform = get_transform(name, **params)

        transform_list.append(transform)

    # Fix for PairedTransform
    if len(transform_list) == 1:
        fn = transform_list[0]
        if hasattr(fn, "__call__") and fn.__call__.__code__.co_argcount >= 2:
            return fn

    return transforms.Compose(transform_list)
