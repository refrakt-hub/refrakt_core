"""
Transform utilities for Refrakt.

This module contains internal helper functions used by the transform builder.
"""

from typing import Any, Callable, Dict, List, Union

from omegaconf import ListConfig
from refrakt_core.registry.transform_registry import get_transform
from torchvision import transforms


def _resolve_transform_sequence(cfg: Union[List[Any], ListConfig, Dict[str, Any]]) -> Union[List[Any], ListConfig]:
    """Resolve the actual transform sequence from configuration."""
    if isinstance(cfg, (list, ListConfig)):
        return cfg
    elif isinstance(cfg, dict):
        if "views" in cfg:
            return cfg["views"][0]
        elif "components" in cfg:
            return cfg["components"]
        else:
            raise ValueError(f"Unsupported transform configuration format: {cfg}")
    else:
        raise TypeError(f"Invalid transform config type: {type(cfg)}")


def _build_nested_transform(name: str, params: Dict[str, Any], build_transform_fn: Callable) -> Callable:
    """Build nested transform blocks like RandomApply."""
    nested_cfgs = params.get("transforms", [])
    nested_transforms = build_transform_fn(nested_cfgs)

    if isinstance(nested_transforms, transforms.Compose):
        nested_transforms = nested_transforms.transforms

    return get_transform(
        "RandomApply", nested_transforms, p=params.get("p", 0.5)
    )


def _build_simple_transform(name: str, params: Dict[str, Any]) -> Callable:
    """Build simple transform from registry."""
    return get_transform(name, **params)


def _build_transform_list(transform_sequence: Union[List[Any], ListConfig], build_transform_fn: Callable) -> List[Callable]:
    """Build list of transforms from sequence."""
    transform_list = []
    
    for t in transform_sequence:
        name = t["name"]
        params = t.get("params", {})

        # Handle nested transform blocks like RandomApply
        if name.lower() == "randomapply":
            transform = _build_nested_transform(name, params, build_transform_fn)
        else:
            transform = _build_simple_transform(name, params)

        transform_list.append(transform)
    
    return transform_list


def _create_final_transform(transform_list: List[Callable]) -> Callable:
    """Create the final transform from the list."""
    # Fix for PairedTransform
    if len(transform_list) == 1:
        fn = transform_list[0]
        if hasattr(fn, "__call__") and fn.__call__.__code__.co_argcount >= 2:
            return fn

    return transforms.Compose(transform_list)