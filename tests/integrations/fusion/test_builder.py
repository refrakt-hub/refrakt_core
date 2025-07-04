"""
Comprehensive tests for fusion builder module.
"""

import pytest
from unittest.mock import Mock, patch

from refrakt_core.integrations.fusion.builder import build_fusion_head
from refrakt_core.integrations.cpu.wrapper import SklearnWrapper


def test_build_sklearn_fusion_head_smoke():
    """Smoke test: Build sklearn fusion head successfully."""
    cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_head = build_fusion_head(cfg)
    
    assert isinstance(fusion_head, SklearnWrapper)
    assert fusion_head.model.__class__.__name__ == "RandomForestClassifier"


def test_build_cuml_fusion_head_smoke():
    """Smoke test: Build cuML fusion head successfully."""
    cfg = {
        "type": "cuml",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    # Mock cuML import
    with patch('refrakt_core.integrations.fusion.builder.CuMLWrapper') as mock_cuml:
        mock_cuml.return_value = Mock()
        fusion_head = build_fusion_head(cfg)
        
        assert mock_cuml.called


def test_build_fusion_head_with_fusion_config_sanity():
    """Sanity test: Build fusion head with fusion_head configuration."""
    cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {
            "n_estimators": 5,
            "fusion_head": {"test": "config"}
        }
    }
    
    fusion_head = build_fusion_head(cfg)
    print(f"DEBUG: fusion_head type: {type(fusion_head)}, repr: {repr(fusion_head)}")
    print(f"DEBUG: dir(fusion_head): {dir(fusion_head)}")
    print(f"DEBUG: wrapper_config: {getattr(fusion_head, 'wrapper_config', None)}")
    
    assert isinstance(fusion_head, SklearnWrapper)
    assert fusion_head.wrapper_config.get("fusion_head") == {"test": "config"}


def test_build_fusion_head_with_none_fusion_config_sanity():
    """Sanity test: Build fusion head with None fusion_head configuration."""
    cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {
            "n_estimators": 5,
            "fusion_head": None
        }
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
        "params": {
            "n_estimators": 5,
            "fusion_head": {"path": "/nonexistent/path"}
        }
    }
    
    # Should fall back to creating new wrapper when path doesn't exist
    fusion_head = build_fusion_head(cfg)
    print(f"DEBUG: fusion_head type: {type(fusion_head)}, repr: {repr(fusion_head)}")
    print(f"DEBUG: dir(fusion_head): {dir(fusion_head)}")
    print(f"DEBUG: wrapper_config: {getattr(fusion_head, 'wrapper_config', None)}")
    
    assert isinstance(fusion_head, SklearnWrapper)


def test_build_fusion_head_with_empty_params_sanity():
    """Sanity test: Build fusion head with empty params."""
    cfg = {
        "type": "sklearn",
        "model": "random_forest"
    }
    
    fusion_head = build_fusion_head(cfg)
    
    assert isinstance(fusion_head, SklearnWrapper)


def test_build_fusion_head_with_full_class_path_sanity():
    """Sanity test: Build fusion head with full class path."""
    cfg = {
        "type": "sklearn",
        "model": "sklearn.ensemble.RandomForestClassifier",
        "params": {"n_estimators": 5}
    }
    
    fusion_head = build_fusion_head(cfg)
    
    assert isinstance(fusion_head, SklearnWrapper)


def test_unsupported_fusion_head_type_unit():
    """Unit test: Verify error for unsupported fusion head type."""
    cfg = {
        "type": "unsupported_type",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    with pytest.raises(ValueError, match="Unsupported fusion head type"):
        build_fusion_head(cfg)


def test_missing_type_key_unit():
    """Unit test: Verify error for missing type key."""
    cfg = {
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    with pytest.raises(KeyError):
        build_fusion_head(cfg)


def test_missing_model_key_unit():
    """Unit test: Verify error for missing model key."""
    cfg = {
        "type": "sklearn",
        "params": {"n_estimators": 5}
    }
    
    with pytest.raises(KeyError):
        build_fusion_head(cfg)


def test_case_insensitive_type_unit():
    """Unit test: Verify case insensitive type handling."""
    cfg = {
        "type": "SKLEARN",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_head = build_fusion_head(cfg)
    
    assert isinstance(fusion_head, SklearnWrapper)


def test_fusion_head_parameter_isolation_unit():
    """Unit test: Verify fusion_head parameter is isolated from model params."""
    cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {
            "n_estimators": 5,
            "fusion_head": {"test": "value"}
        }
    }
    
    fusion_head = build_fusion_head(cfg)
    
    # fusion_head should be extracted and not passed to model
    assert fusion_head.wrapper_config.get("fusion_head") == {"test": "value"}
    # Model should still be created with n_estimators=5
    assert fusion_head.model.n_estimators == 5


def test_multiple_fusion_head_types_unit():
    """Unit test: Test multiple fusion head types."""
    # Test sklearn
    sklearn_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    sklearn_head = build_fusion_head(sklearn_cfg)
    assert isinstance(sklearn_head, SklearnWrapper)
    
    # Test cuml (with mock)
    with patch('refrakt_core.integrations.fusion.builder.CuMLWrapper') as mock_cuml:
        mock_cuml.return_value = Mock()
        cuml_cfg = {
            "type": "cuml",
            "model": "random_forest",
            "params": {"n_estimators": 5}
        }
        cuml_head = build_fusion_head(cuml_cfg)
        assert mock_cuml.called


def test_fusion_head_config_copy_unit():
    """Unit test: Verify fusion head config is properly copied."""
    original_params = {
        "n_estimators": 5,
        "fusion_head": {"test": "original"}
    }
    
    cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": original_params.copy()
    }
    
    fusion_head = build_fusion_head(cfg)
    
    # Original params should not be modified
    assert original_params["fusion_head"] == {"test": "original"}
    # fusion_head should be extracted
    assert fusion_head.wrapper_config.get("fusion_head") == {"test": "original"} 