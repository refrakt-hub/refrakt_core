"""
Model registry loader for cuml-based fusion heads.

This module provides a loader function that reads a YAML file
containing a mapping from user-friendly model names to fully qualified
Python class paths (e.g., "random_forest" → "cuml.ensemble.RandomForestClassifier").

This registry enables shorthand model specification in config files
while preserving dynamic import capability.
"""

import importlib.resources as pkg_resources
from typing import Dict, cast

import yaml


def load_cuml_registry() -> Dict[str, str]:
    """
    Load the cuml model registry from a YAML file.

    Returns:
        Dict[str, str]: A dictionary mapping short model keys
                        to fully qualified cuml class paths.

    Raises:
        FileNotFoundError: If the registry YAML file is missing.
        yaml.YAMLError: If the YAML file is malformed.

    Example:
        >>> registry = load_cuml_registry()
        >>> "random_forest" in registry
        True
        >>> registry["svc"]
        'cuml.svm.SVC'
    """
    with pkg_resources.files("refrakt_core.integrations.registry").joinpath(
        "cuml_registry.yaml"
    ).open("r") as f:
        content = cast(Dict[str, str], yaml.safe_load(f))
    return content
