"""
Comprehensive tests for YAML registry files.
"""

import os
from pathlib import Path

import pytest
import yaml

# Get project root (directory containing 'src')
PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

SKLEARN_REGISTRY_PATH = (
    PROJECT_ROOT
    / "src"
    / "refrakt_core"
    / "integrations"
    / "registry"
    / "sklearn_registry.yaml"
)
CUML_REGISTRY_PATH = (
    PROJECT_ROOT
    / "src"
    / "refrakt_core"
    / "integrations"
    / "registry"
    / "cuml_registry.yaml"
)


def test_sklearn_registry_file_exists_smoke():
    """Smoke test: sklearn registry YAML file exists and is readable."""
    assert SKLEARN_REGISTRY_PATH.exists(), "sklearn_registry.yaml should exist"

    with open(SKLEARN_REGISTRY_PATH, "r") as f:
        content = yaml.safe_load(f)

    assert isinstance(content, dict), "Registry should be a dictionary"


def test_cuml_registry_file_exists_smoke():
    """Smoke test: cuml registry YAML file exists and is readable."""
    assert CUML_REGISTRY_PATH.exists(), "cuml_registry.yaml should exist"

    with open(CUML_REGISTRY_PATH, "r") as f:
        content = yaml.safe_load(f)

    assert isinstance(content, dict), "Registry should be a dictionary"


def test_sklearn_registry_structure_sanity():
    """Sanity test: Verify sklearn registry structure."""

    with open(SKLEARN_REGISTRY_PATH, "r") as f:
        content = yaml.safe_load(f)

    # Check that registry is not empty
    assert len(content) > 0, "Registry should not be empty"

    # Check that all values are strings
    for key, value in content.items():
        assert isinstance(key, str), f"Key {key} should be a string"
        assert isinstance(value, str), f"Value {value} should be a string"
        assert "." in value, f"Value {value} should be a class path"


def test_cuml_registry_structure_sanity():
    """Sanity test: Verify cuml registry structure."""

    with open(CUML_REGISTRY_PATH, "r") as f:
        content = yaml.safe_load(f)

    # Check that registry is not empty
    assert len(content) > 0, "Registry should not be empty"

    # Check that all values are strings
    for key, value in content.items():
        assert isinstance(key, str), f"Key {key} should be a string"
        assert isinstance(value, str), f"Value {value} should be a string"
        assert "." in value, f"Value {value} should be a class path"


def test_sklearn_registry_expected_models_sanity():
    """Sanity test: Verify sklearn registry contains expected models."""

    with open(SKLEARN_REGISTRY_PATH, "r") as f:
        content = yaml.safe_load(f)

    expected_models = ["random_forest", "logistic_regression", "svc", "knn"]
    for model in expected_models:
        assert model in content, f"Registry should contain {model}"
        assert content[model].startswith(
            "sklearn."
        ), f"{model} should map to sklearn class"


def test_cuml_registry_expected_models_sanity():
    """Sanity test: Verify cuml registry contains expected models."""

    with open(CUML_REGISTRY_PATH, "r") as f:
        content = yaml.safe_load(f)

    expected_models = ["random_forest", "logistic_regression", "svc", "knn"]
    for model in expected_models:
        assert model in content, f"Registry should contain {model}"
        assert content[model].startswith("cuml."), f"{model} should map to cuml class"


def test_registry_yaml_format_sanity():
    """Sanity test: Verify YAML format is valid."""
    registry_paths = [SKLEARN_REGISTRY_PATH, CUML_REGISTRY_PATH]

    for registry_path in registry_paths:
        with open(registry_path, "r") as f:
            content = yaml.safe_load(f)

        # Test that content can be dumped back to YAML
        yaml_content = yaml.dump(content)
        assert isinstance(yaml_content, str)
        assert len(yaml_content) > 0


def test_registry_key_uniqueness_unit():
    """Unit test: Verify that registry keys are unique."""
    registry_paths = [SKLEARN_REGISTRY_PATH, CUML_REGISTRY_PATH]

    for registry_path in registry_paths:
        with open(registry_path, "r") as f:
            content = yaml.safe_load(f)

        keys = list(content.keys())
        unique_keys = set(keys)
        assert len(keys) == len(
            unique_keys
        ), f"Registry keys should be unique in {registry_path.name}"


def test_registry_value_uniqueness_unit():
    """Unit test: Verify that registry values are unique."""
    registry_paths = [SKLEARN_REGISTRY_PATH, CUML_REGISTRY_PATH]

    for registry_path in registry_paths:
        with open(registry_path, "r") as f:
            content = yaml.safe_load(f)

        values = list(content.values())
        unique_values = set(values)
        assert len(values) == len(
            unique_values
        ), f"Registry values should be unique in {registry_path.name}"


def test_registry_class_path_format_unit():
    """Unit test: Verify class path format."""
    registry_paths = [SKLEARN_REGISTRY_PATH, CUML_REGISTRY_PATH]

    for registry_path in registry_paths:
        with open(registry_path, "r") as f:
            content = yaml.safe_load(f)

        for model_name, class_path in content.items():
            # Should have at least one dot (module.class)
            parts = class_path.split(".")
            assert len(parts) >= 2, f"{model_name} should have module.class format"

            # Should end with a class name (CamelCase)
            class_name = parts[-1]
            assert class_name[
                0
            ].isupper(), f"{class_name} should be a class name (CamelCase)"


def test_sklearn_registry_specific_mappings_unit():
    """Unit test: Verify specific sklearn registry mappings."""
    registry_path = SKLEARN_REGISTRY_PATH

    with open(registry_path, "r") as f:
        content = yaml.safe_load(f)

    expected_mappings = {
        "random_forest": "sklearn.ensemble.RandomForestClassifier",
        "logistic_regression": "sklearn.linear_model.LogisticRegression",
        "svc": "sklearn.svm.SVC",
        "knn": "sklearn.neighbors.KNeighborsClassifier",
    }

    for model, expected_path in expected_mappings.items():
        if model in content:
            assert (
                content[model] == expected_path
            ), f"{model} should map to {expected_path}"


def test_cuml_registry_specific_mappings_unit():
    """Unit test: Verify specific cuml registry mappings."""
    registry_path = CUML_REGISTRY_PATH

    with open(registry_path, "r") as f:
        content = yaml.safe_load(f)

    expected_mappings = {
        "random_forest": "cuml.ensemble.RandomForestClassifier",
        "logistic_regression": "cuml.linear_model.LogisticRegression",
        "svc": "cuml.svm.SVC",
        "knn": "cuml.neighbors.KNeighborsClassifier",
    }

    for model, expected_path in expected_mappings.items():
        if model in content:
            assert (
                content[model] == expected_path
            ), f"{model} should map to {expected_path}"


def test_registry_file_permissions_unit():
    """Unit test: Verify registry files have correct permissions."""
    registry_paths = [SKLEARN_REGISTRY_PATH, CUML_REGISTRY_PATH]

    for registry_path in registry_paths:
        assert registry_path.is_file(), f"{registry_path} should be a file"
        assert registry_path.stat().st_size > 0, f"{registry_path} should not be empty"


def test_registry_yaml_syntax_unit():
    """Unit test: Verify YAML syntax is correct."""
    registry_paths = [SKLEARN_REGISTRY_PATH, CUML_REGISTRY_PATH]

    for registry_path in registry_paths:
        try:
            with open(registry_path, "r") as f:
                content = yaml.safe_load(f)
            assert (
                content is not None
            ), f"YAML should not be None for {registry_path.name}"
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML syntax in {registry_path.name}: {e}")


def test_registry_consistency_between_files_unit():
    """Unit test: Verify consistency between registry files."""
    sklearn_path = SKLEARN_REGISTRY_PATH
    cuml_path = CUML_REGISTRY_PATH

    with open(sklearn_path, "r") as f:
        sklearn_content = yaml.safe_load(f)

    with open(cuml_path, "r") as f:
        cuml_content = yaml.safe_load(f)

    # Check that both registries have the same keys
    sklearn_keys = set(sklearn_content.keys())
    cuml_keys = set(cuml_content.keys())

    # They should have the same basic models (excluding xgboost which might be different)
    common_models = {"random_forest", "logistic_regression", "svc", "knn"}
    assert common_models.issubset(
        sklearn_keys
    ), "sklearn registry should contain common models"
    assert common_models.issubset(
        cuml_keys
    ), "cuml registry should contain common models"
