"""
Comprehensive tests for feature engineering module.
"""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from refrakt_core.integrations.ml.feature_engineering import build_feature_pipeline


def generate_dummy_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 10)


def test_build_feature_pipeline_smoke():
    """Smoke test: Build a simple feature pipeline."""
    steps = [
        {"name": "standard_scaler", "params": {}},
        {"name": "pca", "params": {"n_components": 5}},
    ]
    pipeline = build_feature_pipeline(steps)
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2


def test_build_feature_pipeline_with_params_smoke():
    """Smoke test: Build pipeline with custom parameters."""
    steps = [
        {"name": "minmax_scaler", "params": {"feature_range": (0, 1)}},
        {"name": "pca", "params": {"n_components": 3}},
    ]
    pipeline = build_feature_pipeline(steps)
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2


def test_build_feature_pipeline_empty_steps_smoke():
    """Smoke test: Build pipeline with empty steps list."""
    steps = []
    pipeline = build_feature_pipeline(steps)
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 0


def test_pipeline_fit_transform_sanity():
    """Sanity test: Pipeline can fit and transform data."""
    X = generate_dummy_data()
    steps = [
        {"name": "standard_scaler", "params": {}},
        {"name": "pca", "params": {"n_components": 5}},
    ]
    pipeline = build_feature_pipeline(steps)

    # Fit and transform
    X_transformed = pipeline.fit_transform(X)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] == 5  # PCA components


def test_pipeline_fit_predict_sanity():
    """Sanity test: Pipeline can fit and then transform new data."""
    X_train = generate_dummy_data()
    X_test = generate_dummy_data()[:20]  # Smaller test set

    steps = [
        {"name": "standard_scaler", "params": {}},
        {"name": "pca", "params": {"n_components": 3}},
    ]
    pipeline = build_feature_pipeline(steps)

    # Fit on training data
    pipeline.fit(X_train)

    # Transform test data
    X_test_transformed = pipeline.transform(X_test)
    assert X_test_transformed.shape[0] == X_test.shape[0]
    assert X_test_transformed.shape[1] == 3


def test_pipeline_steps_order_sanity():
    """Sanity test: Pipeline steps are in correct order."""
    steps = [
        {"name": "standard_scaler", "params": {}},
        {"name": "minmax_scaler", "params": {}},
        {"name": "pca", "params": {"n_components": 2}},
    ]
    pipeline = build_feature_pipeline(steps)

    assert len(pipeline.steps) == 3
    assert pipeline.steps[0][0] == "standard_scaler"
    assert pipeline.steps[1][0] == "minmax_scaler"
    assert pipeline.steps[2][0] == "pca"


def test_pipeline_without_params_sanity():
    """Sanity test: Pipeline works with steps that have no params."""
    steps = [
        {"name": "standard_scaler"},
        {"name": "pca", "params": {"n_components": 4}},
    ]
    pipeline = build_feature_pipeline(steps)
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2


def test_unknown_transformer_error_unit():
    """Unit test: Verify error handling for unknown transformers."""
    steps = [{"name": "unknown_transformer", "params": {}}]
    with pytest.raises(ValueError, match="Unknown feature transformer"):
        build_feature_pipeline(steps)


def test_missing_name_key_unit():
    """Unit test: Verify error handling for missing 'name' key."""
    steps = [{"params": {"n_components": 5}}]  # Missing 'name'
    with pytest.raises(KeyError):
        build_feature_pipeline(steps)


def test_invalid_params_type_unit():
    """Unit test: Verify error handling for invalid params type."""
    steps = [{"name": "standard_scaler", "params": "invalid_params"}]
    with pytest.raises(TypeError):
        build_feature_pipeline(steps)


def test_empty_params_dict_unit():
    """Unit test: Verify handling of empty params dictionary."""
    steps = [{"name": "standard_scaler", "params": {}}]
    pipeline = build_feature_pipeline(steps)
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 1


def test_none_params_unit():
    """Unit test: Verify handling of None params."""
    steps = [{"name": "standard_scaler", "params": None}]
    with pytest.raises(TypeError):
        build_feature_pipeline(steps)


def test_pipeline_attributes_unit():
    """Unit test: Verify pipeline has expected attributes."""
    steps = [
        {"name": "standard_scaler", "params": {}},
        {"name": "pca", "params": {"n_components": 3}},
    ]
    pipeline = build_feature_pipeline(steps)

    assert hasattr(pipeline, "steps")
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "transform")
    assert hasattr(pipeline, "fit_transform")


def test_pipeline_step_names_unit():
    """Unit test: Verify pipeline step names are correct."""
    steps = [
        {"name": "standard_scaler", "params": {}},
        {"name": "pca", "params": {"n_components": 2}},
    ]
    pipeline = build_feature_pipeline(steps)

    step_names = [step[0] for step in pipeline.steps]
    assert step_names == ["standard_scaler", "pca"]


def test_pipeline_step_types_unit():
    """Unit test: Verify pipeline step types are correct."""
    steps = [
        {"name": "standard_scaler", "params": {}},
        {"name": "pca", "params": {"n_components": 2}},
    ]
    pipeline = build_feature_pipeline(steps)

    assert isinstance(pipeline.steps[0][1], StandardScaler)
    assert isinstance(pipeline.steps[1][1], PCA)
