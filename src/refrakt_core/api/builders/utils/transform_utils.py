"""
Transform utilities for Refrakt.

This module provides utility functions for transform building and composition \
    operations, extracted from the main transform_builder to reduce complexity \
    and improve maintainability.

The module handles:
- Transform configuration parsing and validation
- Transform sequence resolution from various configuration formats
- Nested transform block construction (e.g., RandomApply)
- Simple transform instantiation from registry
- Transform list building and composition
- Final transform creation with special handling for paired transforms

These utilities support various transform architectures including simple transforms,
nested transform blocks, and complex transform sequences for data augmentation
and preprocessing pipelines.

Typical usage involves passing configuration dictionaries and transform registries
to build transform pipelines for data loading and preprocessing.
"""

import inspect
from typing import Any, Callable, Dict, List, TypeVar, Union, cast

from omegaconf import ListConfig
from torchvision import transforms  # type: ignore[import-untyped]

from refrakt_core.registry.transform_registry import get_transform

T = TypeVar("T")


def _resolve_transform_sequence(
    cfg: Union[List[Any], ListConfig, Dict[str, Any]],
) -> Union[List[Any], ListConfig]:
    """
    Resolve the actual transform sequence from configuration.

    This function handles different configuration formats for transform sequences.
    It supports direct lists, ListConfig objects, and dictionary formats with
    'views' or 'components' keys for multi-view or component-based transforms.

    Args:
        cfg: Configuration object that may be a list, ListConfig, or dictionary
             containing transform sequence specifications

    Returns:
        Resolved transform sequence as a list or ListConfig

    Raises:
        ValueError: If the configuration format is unsupported or missing required keys
        TypeError: If the configuration type is invalid
    """
    if isinstance(cfg, (list, ListConfig)):
        return cfg
    if isinstance(cfg, dict):
        if "views" in cfg:
            return cast(Union[List[Any], ListConfig], cfg["views"][0])
        if "components" in cfg:
            return cast(Union[List[Any], ListConfig], cfg["components"])
        raise ValueError(f"Unsupported transform configuration format: {cfg}")
    raise TypeError(f"Invalid transform config type: {type(cfg)}")


def _build_nested_transform(
    name: str, params: Dict[str, Any], build_transform_fn: Callable[..., Any]
) -> Callable[..., Any]:
    """
    Build nested transform blocks like RandomApply.

    This function constructs complex transform blocks that contain other transforms
    as nested components. It handles the special case of transforms like RandomApply
    that need to compose multiple sub-transforms.

    Args:
        name: Name of the nested transform (e.g., "RandomApply")
        params: Parameters dictionary containing nested transform configurations
        build_transform_fn: Function to build the nested transforms

    Returns:
        Nested transform function with composed sub-transforms

    Note:
        This function is specifically designed for transforms that contain
        other transforms as parameters, such as RandomApply with a list of
        transforms to apply randomly.
    """
    nested_cfgs = params.get("transforms", [])
    nested_transforms = build_transform_fn(nested_cfgs)

    # Ensure nested_transforms is a list
    if not isinstance(nested_transforms, list):
        nested_transforms = [nested_transforms]

    if isinstance(nested_transforms, transforms.Compose):
        nested_transforms = nested_transforms.transforms

    return cast(
        Callable[..., Any],
        get_transform("RandomApply", nested_transforms, p=params.get("p", 0.5)),
    )


def _build_simple_transform(name: str, params: Dict[str, Any]) -> Callable[..., Any]:
    """
    Build simple transform from registry.

    This function instantiates a single transform from the transform registry
    using the provided name and parameters.

    Args:
        name: Name of the transform to instantiate from the registry
        params: Transform-specific parameters dictionary

    Returns:
        Transform function from the registry with applied parameters
    """
    return cast(Callable[..., Any], get_transform(name, **params))


def _build_transform_list(
    transform_sequence: Union[List[Any], ListConfig],
    build_transform_fn: Callable[..., Any],
) -> List[Callable[..., Any]]:
    """
    Build list of transforms from sequence configuration.

    This function processes a sequence of transform configurations and builds
    the corresponding transform functions. It handles both simple transforms
    and nested transform blocks like RandomApply.

    Args:
        transform_sequence: List or ListConfig containing transform configurations
        build_transform_fn: Function to build transforms from configurations

    Returns:
        List of transform functions ready for composition

    Note:
        The function automatically detects and handles nested transform blocks
        by checking if the transform name is "RandomApply" (case-insensitive).
    """
    transform_list: List[Callable[..., Any]] = []

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


def _create_final_transform(
    transform_list: List[Callable[..., Any]],
) -> Callable[..., Any]:
    """
    Create the final transform from the list of transforms.

    This function composes the final transform pipeline from a list of
    individual transforms. It includes special handling for paired transforms
    that expect multiple inputs.

    Args:
        transform_list: List of transform functions to compose

    Returns:
        Final composed transform function or single transform if appropriate

    Note:
        Special handling is provided for paired transforms that expect
        multiple arguments, avoiding unnecessary composition when only
        one transform is present.
    """
    # Fix for PairedTransform
    if len(transform_list) == 1:
        fn = transform_list[0]
        if callable(fn):
            try:
                sig = inspect.signature(fn)
                if len(sig.parameters) >= 2:
                    return fn
            except (ValueError, TypeError):
                # Could not inspect, skip or handle as needed
                pass

    return cast(Callable[..., Any], transforms.Compose(transform_list))
