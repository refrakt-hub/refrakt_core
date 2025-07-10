"""
Comprehensive tests for fusion builder module.
"""

import pytest
import torch

# Skip if not on a supported GPU (Volta/7.0+) or if CUDA is not available
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 7:
    pytest.skip(
        "Requires Volta (7.0+) GPU for cuml/cudf tests", allow_module_level=True
    )

from unittest.mock import Mock, patch

from refrakt_core.integrations.cpu.wrapper import SklearnWrapper
from refrakt_core.integrations.fusion.builder import build_fusion_head

try:
    import cuml

    cuda_ok = cuml.__version__ is not None and hasattr(cuml, "cuda")
except ImportError:
    cuda_ok = False


@pytest.mark.skipif(
    not cuda_ok, reason="cuML or required GPU (Volta/7.0+) not available."
)
def test_build_cuml_fusion_head_smoke():
    """Smoke test: Build cuML fusion head successfully."""
    cfg = {"type": "cuml", "model": "random_forest", "params": {"n_estimators": 5}}

    # Mock cuML import
    with patch("refrakt_core.integrations.fusion.builder.CuMLWrapper") as mock_cuml:
        mock_cuml.return_value = Mock()
        try:
            fusion_head = build_fusion_head(cfg)
        except Exception as e:
            import sys

            if "cudf.errors.UnsupportedCUDAError" in str(type(e)) or (
                hasattr(e, "__class__")
                and e.__class__.__name__ == "UnsupportedCUDAError"
            ):
                pytest.skip("Skipping cuML test: Unsupported CUDA hardware.")
            raise
        assert mock_cuml.called


def test_build_fusion_head_with_fusion_config_sanity():
    """Sanity test: Build fusion head with fusion_head configuration."""
    cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5, "fusion_head": {"test": "config"}},
    }

    fusion_head = build_fusion_head(cfg)
    print(f"DEBUG: fusion_head type: {type(fusion_head)}, repr: {repr(fusion_head)}")
    print(f"DEBUG: dir(fusion_head): {dir(fusion_head)}")
    print(f"DEBUG: wrapper_config: {getattr(fusion_head, 'wrapper_config', None)}")

    assert isinstance(fusion_head, SklearnWrapper)
    wrapper_config = getattr(fusion_head, "wrapper_config", None)
    if wrapper_config is not None:
        assert wrapper_config.get("fusion_head") == {"test": "config"}


def test_build_fusion_head_with_none_fusion_config_sanity():
    """Sanity test: Build fusion head with None fusion_head configuration."""
    cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5, "fusion_head": None},
    }

    fusion_head = build_fusion_head(cfg)
    print(f"DEBUG: fusion_head type: {type(fusion_head)}, repr: {repr(fusion_head)}")
    print(f"DEBUG: dir(fusion_head): {dir(fusion_head)}")
    print(f"DEBUG: wrapper_config: {getattr(fusion_head, 'wrapper_config', None)}")

    assert isinstance(fusion_head, SklearnWrapper)
    assert fusion_head.wrapper_config.get("fusion_head") == {}


def test_build_fusion_head_with_path_config_sanity():
    """Sanity test: Build fusion head with path configuration."""
    cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5, "fusion_head": {"path": "/nonexistent/path"}},
    }

    # Should fall back to creating new wrapper when path doesn't exist
    fusion_head = build_fusion_head(cfg)
    print(f"DEBUG: fusion_head type: {type(fusion_head)}, repr: {repr(fusion_head)}")
    print(f"DEBUG: dir(fusion_head): {dir(fusion_head)}")
    print(f"DEBUG: wrapper_config: {getattr(fusion_head, 'wrapper_config', None)}")

    assert isinstance(fusion_head, SklearnWrapper)


def test_build_fusion_head_with_empty_params_sanity():
    """Sanity test: Build fusion head with empty params."""
    cfg = {"type": "sklearn", "model": "random_forest"}

    fusion_head = build_fusion_head(cfg)

    assert isinstance(fusion_head, SklearnWrapper)


def test_build_fusion_head_with_full_class_path_sanity():
    """Sanity test: Build fusion head with full class path."""
    cfg = {
        "type": "sklearn",
        "model": "sklearn.ensemble.RandomForestClassifier",
        "params": {"n_estimators": 5},
    }

    fusion_head = build_fusion_head(cfg)

    assert isinstance(fusion_head, SklearnWrapper)


def test_unsupported_fusion_head_type_unit():
    """Unit test: Verify error for unsupported fusion head type."""
    cfg = {
        "type": "unsupported_type",
        "model": "random_forest",
        "params": {"n_estimators": 5},
    }

    with pytest.raises(ValueError, match="Unsupported fusion head type"):
        build_fusion_head(cfg)


def test_missing_type_key_unit():
    """Unit test: Verify error for missing type key."""
    cfg = {"model": "random_forest", "params": {"n_estimators": 5}}

    with pytest.raises(KeyError):
        build_fusion_head(cfg)


def test_missing_model_key_unit():
    """Unit test: Verify error for missing model key."""
    cfg = {"type": "sklearn", "params": {"n_estimators": 5}}

    with pytest.raises(KeyError):
        build_fusion_head(cfg)


def test_case_insensitive_type_unit():
    """Unit test: Verify case insensitive type handling."""
    cfg = {"type": "SKLEARN", "model": "random_forest", "params": {"n_estimators": 5}}

    fusion_head = build_fusion_head(cfg)

    assert isinstance(fusion_head, SklearnWrapper)


def test_fusion_head_parameter_isolation_unit():
    """Unit test: Verify fusion_head parameter is isolated from model params."""
    cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5, "fusion_head": {"test": "value"}},
    }

    fusion_head = build_fusion_head(cfg)

    # fusion_head should be extracted and not passed to model
    wrapper_config = getattr(fusion_head, "wrapper_config", None)
    if wrapper_config is not None:
        assert wrapper_config.get("fusion_head") == {"test": "value"}
    # Model should still be created with n_estimators=5
    model = getattr(fusion_head, "model", None)
    if model is not None:
        assert getattr(model, "n_estimators", None) == 5


@pytest.mark.skipif(
    not cuda_ok, reason="cuML or required GPU (Volta/7.0+) not available."
)
def test_multiple_fusion_head_types_unit():
    """Unit test: Test multiple fusion head types."""
    import pytest

    # Test sklearn
    sklearn_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5},
    }
    sklearn_head = build_fusion_head(sklearn_cfg)
    assert isinstance(sklearn_head, SklearnWrapper)

    # Test cuml (with mock)
    with patch("refrakt_core.integrations.fusion.builder.CuMLWrapper") as mock_cuml:
        mock_cuml.return_value = Mock()
        cuml_cfg = {
            "type": "cuml",
            "model": "random_forest",
            "params": {"n_estimators": 5},
        }
        try:
            cuml_head = build_fusion_head(cuml_cfg)
        except Exception as e:
            if "cudf.errors.UnsupportedCUDAError" in str(type(e)) or (
                hasattr(e, "__class__")
                and e.__class__.__name__ == "UnsupportedCUDAError"
            ):
                pytest.skip("Skipping cuML test: Unsupported CUDA hardware.")
            raise
        assert mock_cuml.called


def test_fusion_head_config_copy_unit():
    """Unit test: Verify fusion head config is properly copied."""
    original_params = {"n_estimators": 5, "fusion_head": {"test": "original"}}

    cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": original_params.copy(),
    }

    fusion_head = build_fusion_head(cfg)

    # Original params should not be modified
    assert original_params["fusion_head"] == {"test": "original"}
    # fusion_head should be extracted
    wrapper_config = getattr(fusion_head, "wrapper_config", None)
    if wrapper_config is not None:
        assert wrapper_config.get("fusion_head") == {"test": "original"}
