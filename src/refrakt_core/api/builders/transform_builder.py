"""
Transform builder for Refrakt.

This module provides utilities to construct data transformation pipelines \
    from configuration dictionaries. It supports both simple and nested \
    transforms, including custom and registry-based transforms, and integrates \
    with PyTorch Compose.

The module handles:
- Transform configuration validation and parsing
- Simple and nested transform sequence building
- RandomApply and PairedTransform support
- Registry-based transform discovery and instantiation
- PyTorch Compose integration for transform pipelines
- Recursive transform building for complex configurations
- Custom transform integration and parameter passing

Typical usage involves passing a configuration (list, dict, or ListConfig) \
    sdescribing the transform sequence or components.
"""

from typing import Any, Callable, Dict, List, Union

from omegaconf import ListConfig

from refrakt_core.api.builders.utils.transform_utils import (
    _build_transform_list,
    _create_final_transform,
    _resolve_transform_sequence,
)


def build_transform(
    cfg: Union[List[Any], ListConfig, Dict[str, Any]],
) -> Callable[..., Any]:
    """
    Build a transform pipeline from configuration using the transform registry.

    This function supports both flat and nested transform configurations, including
    RandomApply and PairedTransform. It recursively builds and composes transforms
    as needed for flexible data preprocessing.

    The function handles various configuration formats:
    - List: Direct sequence of transforms
    - Dict with 'views': Multi-view transform configuration
    - Dict with 'components': Multi-component transform configuration
    - Nested structures: Recursive transform building

    Args:
        cfg: Configuration specifying the transform sequence or components.
             Can be a list, ListConfig, or dict with 'views' or 'components' keys.
             Supports nested configurations for complex transform pipelines.

    Returns:
        A composed transform function or callable object that can be applied to data.
        The exact type depends on the transform configuration.

    Raises:
        TypeError: If the configuration is not of a supported type
        ValueError: If the configuration format is not recognized or supported
    """
    transform_sequence = _resolve_transform_sequence(cfg)
    transform_list = _build_transform_list(transform_sequence, build_transform)
    return _create_final_transform(transform_list)
