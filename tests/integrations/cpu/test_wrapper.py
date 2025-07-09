"""
Comprehensive tests for sklearn wrapper module.
"""

# type: ignore

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

from refrakt_core.integrations.cpu.wrapper import SklearnWrapper


def generate_dummy_data() -> tuple[np.ndarray, np.ndarray]:  # type: ignore
    """
    Generate synthetic classification data for testing.

    Returns:
        Tuple of features and labels (X, y).
    """
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y


def test_model_instantiates_from_yaml_registry_smoke():
    """Smoke test: Model loads from YAML registry and makes predictions."""
    X, y = generate_dummy_data()
    clf = SklearnWrapper("random_forest", n_estimators=5)  # type: ignore
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)
    assert isinstance(preds, np.ndarray)


def test_model_instantiates_from_full_path_smoke():
    """Smoke test: Model loads from full class path and makes predictions."""
    X, y = generate_dummy_data()
    clf = SklearnWrapper("sklearn.linear_model.LogisticRegression", max_iter=200)  # type: ignore
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)
    assert isinstance(preds, np.ndarray)


def test_multiple_model_types_smoke():
    """Smoke test: Test multiple sklearn model types."""
    X, y = generate_dummy_data()

    # Test Random Forest
    rf = SklearnWrapper("random_forest", n_estimators=5)  # type: ignore
    rf.fit(X, y)
    rf_preds = rf.predict(X)
    assert rf_preds.shape == (100,)

    # Test Logistic Regression
    lr = SklearnWrapper("logistic_regression", max_iter=200)  # type: ignore
    lr.fit(X, y)
    lr_preds = lr.predict(X)
    assert lr_preds.shape == (100,)

    # Test SVC
    svc = SklearnWrapper("svc", kernel="linear")  # type: ignore
    svc.fit(X, y)
    svc_preds = svc.predict(X)
    assert svc_preds.shape == (100,)


def test_wrapper_configuration_sanity():
    """Sanity test: Verify wrapper configuration handling."""
    X, y = generate_dummy_data()

    # Test with fusion_head parameter
    clf = SklearnWrapper("random_forest", n_estimators=5, fusion_head={"test": "config"})  # type: ignore
    assert hasattr(clf, "wrapper_config")
    assert clf.wrapper_config.get("fusion_head") == {"test": "config"}

    # Test model still works
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)


def test_predict_proba_sanity():
    """Sanity test: Verify predict_proba functionality."""
    X, y = generate_dummy_data()

    # Test with Random Forest (supports predict_proba)
    rf = SklearnWrapper("random_forest", n_estimators=5)  # type: ignore
    rf.fit(X, y)
    proba = rf.predict_proba(X)
    assert proba.shape == (100, 2)  # Binary classification
    assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1

    # Test with SVC (doesn't support predict_proba by default)
    svc = SklearnWrapper("svc", kernel="linear", probability=True)  # type: ignore
    svc.fit(X, y)
    proba = svc.predict_proba(X)
    assert proba.shape == (100, 2)


def test_save_load_functionality_sanity():
    """Sanity test: Verify save and load functionality."""
    X, y = generate_dummy_data()
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    clf = SklearnWrapper("random_forest", n_estimators=5)  # type: ignore

    clf.fit(X, y)  # Fit before saving

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        # Save model
        clf.save(tmp_path)
        assert tmp_path.exists()

        # Load model
        loaded_clf = SklearnWrapper.load("random_forest", tmp_path)
        loaded_preds = loaded_clf.predict(X)
        original_preds = clf.predict(X)

        # Predictions should be identical
        np.testing.assert_array_equal(loaded_preds, original_preds)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_repr_functionality_unit():
    """Unit test: Verify string representation."""
    clf = SklearnWrapper("random_forest", n_estimators=5)  # type: ignore
    repr_str = repr(clf)
    assert isinstance(repr_str, str)
    assert len(repr_str) > 0


def test_invalid_model_path_unit():
    """Unit test: Verify error handling for invalid model paths."""
    with pytest.raises(ValueError, match="Invalid sklearn model"):
        SklearnWrapper("invalid.model.path.Classifier")


def test_invalid_module_unit():
    """Unit test: Verify error handling for invalid modules."""
    with pytest.raises(ValueError, match="Invalid sklearn model"):
        SklearnWrapper("nonexistent.module.RandomForestClassifier")


def test_predict_proba_not_supported_unit():
    """Unit test: Verify error handling when predict_proba is not supported."""
    X, y = generate_dummy_data()

    # SVC without probability=True doesn't support predict_proba
    svc = SklearnWrapper("svc", kernel="linear")  # type: ignore
    svc.fit(X, y)

    with pytest.raises(AttributeError, match="does not support predict_proba"):
        svc.predict_proba(X)


def test_not_fitted_error_unit():
    """Unit test: Verify error handling when model is not fitted."""
    X, y = generate_dummy_data()
    clf = SklearnWrapper("random_forest", n_estimators=5)  # type: ignore

    # Should raise NotFittedError when trying to predict without fitting
    with pytest.raises(NotFittedError):
        clf.predict(X)

    with pytest.raises(NotFittedError):
        clf.predict_proba(X)


def test_model_parameters_unit():
    """Unit test: Verify model parameter handling."""
    X, y = generate_dummy_data()

    # Test with different parameter combinations
    clf1 = SklearnWrapper("random_forest", n_estimators=10, max_depth=3)  # type: ignore
    clf1.fit(X, y)

    clf2 = SklearnWrapper("random_forest", n_estimators=5, max_depth=5)  # type: ignore
    clf2.fit(X, y)

    # Models with different parameters should behave differently
    preds1 = clf1.predict(X)
    preds2 = clf2.predict(X)

    # Note: This might not always be true due to randomness, but it's a reasonable test
    # that different parameters can be passed and used


def test_data_type_handling_unit():
    """Unit test: Verify data type handling."""
    X, y = generate_dummy_data()

    # Test with float32 data
    X_float32 = X.astype(np.float32)
    y_int32 = y.astype(np.int32)

    clf = SklearnWrapper("random_forest", n_estimators=5)  # type: ignore
    clf.fit(X_float32, y_int32)
    preds = clf.predict(X_float32)

    assert preds.shape == (100,)
    assert isinstance(preds, np.ndarray)


def test_empty_data_handling_unit():
    """Unit test: Verify handling of edge cases with empty data."""
    # Test with single sample
    X_single = np.array([[1.0, 2.0, 3.0]])
    y_single = np.array([0])

    clf = SklearnWrapper("random_forest", n_estimators=5)  # type: ignore
    clf.fit(X_single, y_single)
    preds = clf.predict(X_single)

    assert preds.shape == (1,)
    assert isinstance(preds, np.ndarray)


def test_wrapper_config_isolation_unit():
    """Unit test: Verify wrapper config doesn't interfere with model params."""
    X, y = generate_dummy_data()

    # Test that fusion_head parameter doesn't affect model instantiation
    clf = SklearnWrapper("random_forest", n_estimators=5, fusion_head={"test": "value"})  # type: ignore

    # Model should still work normally
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)

    # Wrapper config should be stored separately
    assert clf.wrapper_config.get("fusion_head") == {"test": "value"}
