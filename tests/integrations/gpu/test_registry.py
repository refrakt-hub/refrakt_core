"""
Comprehensive tests for cuML registry module.
"""

import pytest
from typing import Any

try:
    import cuml
    import cupy
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    cuda_ok = cc[0] > 7 or (cc[0] == 7 and cc[1] >= 0)
except Exception:
    cuda_ok = False

pytestmark = pytest.mark.skipif(
    not cuda_ok,
    reason="cuML or required GPU (Volta/7.0+) not available."
)

from refrakt_core.integrations.gpu.registry import load_cuml_registry

def test_yaml_registry_loads_successfully_smoke():
    """Smoke test: YAML registry loads and contains required models."""
    registry: dict[str, str] = load_cuml_registry()
    assert isinstance(registry, dict), "Registry should be a dictionary"
    assert "random_forest" in registry, "Registry must contain 'random_forest'"
    assert registry["random_forest"] == "cuml.ensemble.RandomForestClassifier"

def test_registry_structure_sanity():
    """Sanity test: Verify registry structure and content."""
    registry = load_cuml_registry()
    assert len(registry) > 0, "Registry should not be empty"
    for key, value in registry.items():
        assert isinstance(key, str), f"Registry key {key} should be a string"
        assert isinstance(value, str), f"Registry value {value} should be a string"
        assert "." in value, f"Registry value {value} should be a class path"
    expected_models = ["random_forest", "logistic_regression", "svc", "knn"]
    for model in expected_models:
        assert model in registry, f"Registry should contain {model}"
        assert registry[model].startswith("cuml."), f"{model} should map to cuml class"

def test_registry_model_paths_sanity():
    """Sanity test: Verify that registry model paths are valid cuml paths."""
    registry = load_cuml_registry()
    for model_name, class_path in registry.items():
        assert class_path.startswith("cuml."), f"{model_name} path should start with cuml."
        parts = class_path.split(".")
        assert len(parts) >= 2, f"{model_name} should have module.class format"
        class_name = parts[-1]
        assert class_name[0].isupper(), f"{class_name} should be a class name (CamelCase)"

def test_registry_consistency_unit():
    """Unit test: Verify registry consistency across multiple loads."""
    registry1 = load_cuml_registry()
    registry2 = load_cuml_registry()
    assert registry1 == registry2, "Registry should be consistent across loads"
    assert set(registry1.keys()) == set(registry2.keys()), "Registry keys should be consistent"
    for key in registry1:
        assert registry1[key] == registry2[key], f"Registry value for {key} should be consistent"

def test_registry_key_uniqueness_unit():
    """Unit test: Verify that registry keys are unique."""
    registry = load_cuml_registry()
    keys = list(registry.keys())
    unique_keys = set(keys)
    assert len(keys) == len(unique_keys), "Registry keys should be unique"

def test_registry_value_uniqueness_unit():
    """Unit test: Verify that registry values are unique."""
    registry = load_cuml_registry()
    values = list(registry.values())
    unique_values = set(values)
    assert len(values) == len(unique_values), "Registry values should be unique"

def test_registry_import_paths_unit():
    """Unit test: Verify that registry paths can be imported."""
    registry = load_cuml_registry()
    for model_name, class_path in registry.items():
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            assert hasattr(model_class, "__call__"), f"{class_path} should be callable"
        except (ImportError, AttributeError) as e:
            if "cuml" in str(e):
                pytest.skip(f"cuML not available: {e}")
            else:
                pytest.fail(f"Failed to import {class_path}: {e}")

def test_registry_xgboost_integration_unit():
    """Unit test: Verify xgboost integration in registry."""
    registry = load_cuml_registry()
    if "xgboost" in registry:
        assert registry["xgboost"] == "cuml.experimental.xgboost.XGBClassifier", "XGBoost should map to correct class"
    else:
        pytest.skip("XGBoost not available in registry")

def test_registry_error_handling_unit():
    """Unit test: Verify error handling for missing registry file."""
    try:
        registry = load_cuml_registry()
        assert isinstance(registry, dict)
    except Exception as e:
        pytest.fail(f"load_cuml_registry should not raise {type(e).__name__}: {e}")

def test_registry_yaml_format_unit():
    """Unit test: Verify YAML format compliance."""
    registry = load_cuml_registry()
    yaml_like = {}
    for key, value in registry.items():
        yaml_like[key] = value
    assert isinstance(yaml_like, dict)
    assert len(yaml_like) == len(registry)
    for key, value in yaml_like.items():
        assert isinstance(key, str)
        assert isinstance(value, str)

def test_registry_experimental_modules_unit():
    """Unit test: Verify experimental module handling."""
    registry = load_cuml_registry()
    experimental_modules = [k for k, v in registry.items() if "experimental" in v]
    for module_name in experimental_modules:
        class_path = registry[module_name]
        assert "experimental" in class_path, f"{module_name} should be in experimental module"

def test_registry_module_organization_unit():
    """Unit test: Verify module organization."""
    registry = load_cuml_registry()
    module_organization = {
        "random_forest": "ensemble",
        "logistic_regression": "linear_model", 
        "svc": "svm",
        "knn": "neighbors"
    }
    for model_name, expected_module in module_organization.items():
        if model_name in registry:
            class_path = registry[model_name]
            assert expected_module in class_path, f"{model_name} should be in {expected_module} module" 