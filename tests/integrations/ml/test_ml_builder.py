"""
Comprehensive tests for ml_builder.py.
"""

import numpy as np
import pytest

from refrakt_core.integrations.ml.feature_engineering import build_feature_pipeline
from refrakt_core.integrations.ml.ml_builder import build_ml_pipeline
from refrakt_core.integrations.ml.wrapper import build_ml_model


def generate_dummy_data():
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)
    return X, y


def test_build_ml_pipeline_smoke():
    """Smoke test: Build ML pipeline with minimal config."""
    X, y = generate_dummy_data()
    cfg = {
        "feature_engineering": [
            {"name": "standard_scaler", "params": {}},
            {"name": "pca", "params": {"n_components": 3}},
        ],
        "model": {
            "backend": "sklearn",
            "name": "logistic_regression",
            "params": {"max_iter": 100},
        },
    }
    pipeline, model, X_out, y_out, X_val, y_val = build_ml_pipeline(cfg, X, y)
    assert pipeline is not None
    assert hasattr(pipeline, "fit")
    assert hasattr(model, "fit")
    assert X_out.shape == X.shape
    assert y_out.shape == y.shape
    assert X_val is None
    assert y_val is None


def test_build_ml_pipeline_with_validation_sanity():
    """Sanity test: Build ML pipeline with validation data."""
    X, y = generate_dummy_data()
    X_val, y_val = generate_dummy_data()
    cfg = {
        "feature_engineering": [
            {"name": "minmax_scaler", "params": {"feature_range": (0, 1)}}
        ],
        "model": {
            "backend": "sklearn",
            "name": "random_forest",
            "params": {"n_estimators": 5},
        },
    }
    pipeline, model, X_out, y_out, X_val_out, y_val_out = build_ml_pipeline(
        cfg, X, y, X_val, y_val
    )
    assert pipeline is not None
    assert model is not None
    assert X_val_out is not None
    assert y_val_out is not None
    assert X_val_out.shape == X_val.shape
    assert y_val_out.shape == y_val.shape


def test_build_ml_pipeline_no_feature_engineering_sanity():
    """Sanity test: Build ML pipeline with no feature engineering steps."""
    X, y = generate_dummy_data()
    cfg = {"model": {"backend": "sklearn", "name": "logistic_regression", "params": {}}}
    pipeline, model, X_out, y_out, X_val, y_val = build_ml_pipeline(cfg, X, y)
    assert pipeline is None
    assert model is not None


def test_build_ml_pipeline_invalid_model_unit():
    """Unit test: Error on unknown model backend."""
    X, y = generate_dummy_data()
    cfg = {
        "feature_engineering": [{"name": "standard_scaler", "params": {}}],
        "model": {
            "backend": "unknown_backend",
            "name": "logistic_regression",
            "params": {},
        },
    }
    with pytest.raises(ValueError, match="Unknown ML backend"):
        build_ml_pipeline(cfg, X, y)


def test_build_ml_pipeline_missing_model_key_unit():
    """Unit test: Error on missing model key in config."""
    X, y = generate_dummy_data()
    cfg = {"feature_engineering": [{"name": "standard_scaler", "params": {}}]}
    with pytest.raises(KeyError):
        build_ml_pipeline(cfg, X, y)


def test_build_ml_pipeline_output_shapes_unit():
    """Unit test: Output shapes are correct."""
    X, y = generate_dummy_data()
    cfg = {
        "feature_engineering": [{"name": "standard_scaler", "params": {}}],
        "model": {"backend": "sklearn", "name": "logistic_regression", "params": {}},
    }
    pipeline, model, X_out, y_out, X_val, y_val = build_ml_pipeline(cfg, X, y)
    assert X_out.shape == X.shape
    assert y_out.shape == y.shape


def test_build_ml_pipeline_pipeline_type_unit():
    """Unit test: Pipeline is correct type or None."""
    X, y = generate_dummy_data()
    cfg = {
        "feature_engineering": [],
        "model": {"backend": "sklearn", "name": "logistic_regression", "params": {}},
    }
    pipeline, model, *_ = build_ml_pipeline(cfg, X, y)
    assert pipeline is None or hasattr(pipeline, "fit")
