"""
Transform builder for Refrakt.

This module provides utilities to construct data transformation pipelines from configuration dictionaries.
It supports both simple and nested transforms, including custom and registry-based transforms, and integrates with PyTorch Compose.

Typical usage involves passing a configuration (list, dict, or ListConfig) describing the transform sequence or components.
"""
from typing import Union, List, Any, Callable, Dict
from omegaconf import ListConfig
from refrakt_core.api.builders.utils.transform_utils import (
    _resolve_transform_sequence,
    _build_transform_list,
    _create_final_transform,
)

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
    transform_sequence = _resolve_transform_sequence(cfg)
    transform_list = _build_transform_list(transform_sequence, build_transform)
    return _create_final_transform(transform_list) 