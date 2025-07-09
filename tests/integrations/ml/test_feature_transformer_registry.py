"""
Comprehensive tests for feature transformer registry module.
"""

import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from refrakt_core.integrations.ml.feature_transformer_registry import (
    FEATURE_TRANSFORMER_REGISTRY,
    load_feature_transformer_registry,
)


def test_registry_loads_at_import_smoke():
    """Smoke test: Registry loads successfully at import."""
    from refrakt_core.integrations.ml.feature_transformer_registry import (
        FEATURE_TRANSFORMER_REGISTRY,
    )

    assert isinstance(FEATURE_TRANSFORMER_REGISTRY, dict)


def test_registry_contains_expected_keys_smoke():
    """Smoke test: Registry contains expected transformer keys."""
    expected_keys = {"standard_scaler", "minmax_scaler", "onehot", "pca"}
    actual_keys = set(FEATURE_TRANSFORMER_REGISTRY.keys())
    assert expected_keys.issubset(actual_keys)


def test_registry_values_are_classes_smoke():
    """Smoke test: Registry values are actual classes."""
    for key, value in FEATURE_TRANSFORMER_REGISTRY.items():
        assert hasattr(value, "__call__")  # Should be callable (class)


def test_load_registry_with_valid_yaml_sanity():
    """Sanity test: Load registry with valid YAML content."""
    test_yaml_content = """
standard_scaler: sklearn.preprocessing.StandardScaler
minmax_scaler: sklearn.preprocessing.MinMaxScaler
pca: sklearn.decomposition.PCA
"""

    with patch("builtins.open", mock_open(read_data=test_yaml_content)):
        with patch("yaml.safe_load") as mock_yaml_load:
            mock_yaml_load.return_value = {
                "standard_scaler": "sklearn.preprocessing.StandardScaler",
                "minmax_scaler": "sklearn.preprocessing.MinMaxScaler",
                "pca": "sklearn.decomposition.PCA",
            }

            # Clear registry for testing
            original_registry = FEATURE_TRANSFORMER_REGISTRY.copy()
            FEATURE_TRANSFORMER_REGISTRY.clear()

            try:
                load_feature_transformer_registry()
                assert len(FEATURE_TRANSFORMER_REGISTRY) > 0
            finally:
                # Restore original registry
                FEATURE_TRANSFORMER_REGISTRY.clear()
                FEATURE_TRANSFORMER_REGISTRY.update(original_registry)


def test_registry_imports_correct_modules_sanity():
    """Sanity test: Registry imports correct sklearn modules."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    assert "standard_scaler" in FEATURE_TRANSFORMER_REGISTRY
    assert "minmax_scaler" in FEATURE_TRANSFORMER_REGISTRY
    assert "pca" in FEATURE_TRANSFORMER_REGISTRY

    # Check that the classes are the expected ones
    assert FEATURE_TRANSFORMER_REGISTRY["standard_scaler"] == StandardScaler
    assert FEATURE_TRANSFORMER_REGISTRY["minmax_scaler"] == MinMaxScaler
    assert FEATURE_TRANSFORMER_REGISTRY["pca"] == PCA


def test_registry_classes_are_instantiable_sanity():
    """Sanity test: Registry classes can be instantiated."""
    import numpy as np

    X = np.random.randn(10, 5)

    # Test StandardScaler
    scaler = FEATURE_TRANSFORMER_REGISTRY["standard_scaler"]()
    assert hasattr(scaler, "fit")
    assert hasattr(scaler, "transform")

    # Test PCA
    pca = FEATURE_TRANSFORMER_REGISTRY["pca"](n_components=3)
    assert hasattr(pca, "fit")
    assert hasattr(pca, "transform")


def test_load_registry_with_custom_path_sanity():
    """Sanity test: Load registry with custom YAML path."""
    test_yaml_content = """
custom_scaler: sklearn.preprocessing.StandardScaler
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp_file:
        yaml.dump({"custom_scaler": "sklearn.preprocessing.StandardScaler"}, tmp_file)
        tmp_path = Path(tmp_file.name)

    try:
        # Clear registry for testing
        original_registry = FEATURE_TRANSFORMER_REGISTRY.copy()
        FEATURE_TRANSFORMER_REGISTRY.clear()

        load_feature_transformer_registry(str(tmp_path))
        assert "custom_scaler" in FEATURE_TRANSFORMER_REGISTRY
        assert (
            FEATURE_TRANSFORMER_REGISTRY["custom_scaler"].__name__ == "StandardScaler"
        )
    finally:
        # Restore original registry
        FEATURE_TRANSFORMER_REGISTRY.clear()
        FEATURE_TRANSFORMER_REGISTRY.update(original_registry)
        tmp_path.unlink()


def test_file_not_found_error_unit():
    """Unit test: Verify error handling when YAML file is not found."""
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = False

        with pytest.raises(
            FileNotFoundError, match="feature_transformer_registry.yaml not found"
        ):
            load_feature_transformer_registry()


def test_invalid_yaml_format_unit():
    """Unit test: Verify error handling for invalid YAML format."""
    invalid_yaml_content = """
invalid: yaml: content: here
"""

    with patch("builtins.open", mock_open(read_data=invalid_yaml_content)):
        with pytest.raises(yaml.YAMLError):
            load_feature_transformer_registry()


def test_invalid_import_path_unit():
    """Unit test: Verify error handling for invalid import paths."""
    test_yaml_content = """
invalid_transformer: nonexistent.module.Class
"""

    with patch("builtins.open", mock_open(read_data=test_yaml_content)):
        with patch("yaml.safe_load") as mock_yaml_load:
            mock_yaml_load.return_value = {
                "invalid_transformer": "nonexistent.module.Class"
            }

            with pytest.raises(ModuleNotFoundError):
                load_feature_transformer_registry()


def test_invalid_class_name_unit():
    """Unit test: Verify error handling for invalid class names."""
    test_yaml_content = """
invalid_transformer: sklearn.preprocessing.NonexistentClass
"""

    with patch("builtins.open", mock_open(read_data=test_yaml_content)):
        with patch("yaml.safe_load") as mock_yaml_load:
            mock_yaml_load.return_value = {
                "invalid_transformer": "sklearn.preprocessing.NonexistentClass"
            }

            with pytest.raises(AttributeError):
                load_feature_transformer_registry()


def test_empty_yaml_file_unit():
    """Unit test: Verify handling of empty YAML file."""
    empty_yaml_content = ""

    with patch("builtins.open", mock_open(read_data=empty_yaml_content)):
        with patch("yaml.safe_load") as mock_yaml_load:
            mock_yaml_load.return_value = {}

            # Clear registry for testing
            original_registry = FEATURE_TRANSFORMER_REGISTRY.copy()
            FEATURE_TRANSFORMER_REGISTRY.clear()

            try:
                load_feature_transformer_registry()
                assert len(FEATURE_TRANSFORMER_REGISTRY) == 0
            finally:
                # Restore original registry
                FEATURE_TRANSFORMER_REGISTRY.clear()
                FEATURE_TRANSFORMER_REGISTRY.update(original_registry)


def test_malformed_import_path_unit():
    """Unit test: Verify error handling for malformed import paths."""
    import tempfile

    import yaml

    # Create a temporary YAML file with malformed import path (no dots)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp_file:
        yaml.dump({"malformed": "sklearnpreprocessing"}, tmp_file)  # No dots at all
        tmp_path = Path(tmp_file.name)

    try:
        # Clear registry for testing
        original_registry = FEATURE_TRANSFORMER_REGISTRY.copy()
        FEATURE_TRANSFORMER_REGISTRY.clear()

        with pytest.raises(ValueError, match="not enough values to unpack"):
            load_feature_transformer_registry(str(tmp_path))
    finally:
        # Restore original registry
        FEATURE_TRANSFORMER_REGISTRY.clear()
        FEATURE_TRANSFORMER_REGISTRY.update(original_registry)
        tmp_path.unlink()


def test_registry_persistence_unit():
    """Unit test: Verify registry persists between calls."""
    original_keys = set(FEATURE_TRANSFORMER_REGISTRY.keys())

    # Call load function again
    load_feature_transformer_registry()

    # Registry should still have the same keys
    current_keys = set(FEATURE_TRANSFORMER_REGISTRY.keys())
    assert original_keys == current_keys


def test_registry_immutability_unit():
    """Unit test: Verify registry is not cleared on subsequent loads."""
    original_registry = FEATURE_TRANSFORMER_REGISTRY.copy()

    # Load registry again
    load_feature_transformer_registry()

    # Registry should still contain the same items
    assert len(FEATURE_TRANSFORMER_REGISTRY) == len(original_registry)
    for key in original_registry:
        assert key in FEATURE_TRANSFORMER_REGISTRY
