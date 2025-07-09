"""
Comprehensive tests for cuML wrapper module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Try to import cuML, skip tests if not available
try:
    import cupy as cp

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

import pytest

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
    not cuda_ok, reason="cuML or required GPU (Volta/7.0+) not available."
)

from refrakt_core.integrations.gpu.wrapper import CuMLWrapper


def generate_dummy_data():
    """
    Generate synthetic classification data for testing.

    Returns:
        Tuple of features and labels (X, y).
    """
    if CUML_AVAILABLE:
        X = cp.random.rand(100, 20).astype(cp.float64)
        y = cp.random.randint(0, 2, 100).astype(cp.float64)
    else:
        X = np.random.rand(100, 20).astype(np.float64)
        y = np.random.randint(0, 2, 100).astype(np.float64)
    return X, y


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_model_instantiates_from_yaml_registry_smoke():
    """Smoke test: Model loads from YAML registry and makes predictions."""
    X, y = generate_dummy_data()
    clf = CuMLWrapper("random_forest", n_estimators=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)
    assert isinstance(preds, (np.ndarray, cp.ndarray))


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_model_instantiates_from_full_path_smoke():
    """Smoke test: Model loads from full class path and makes predictions."""
    X, y = generate_dummy_data()
    clf = CuMLWrapper("cuml.linear_model.LogisticRegression", max_iter=200)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)
    assert isinstance(preds, (np.ndarray, cp.ndarray))


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_multiple_model_types_smoke():
    """Smoke test: Test multiple cuML model types."""
    X, y = generate_dummy_data()
    rf = CuMLWrapper("random_forest", n_estimators=5)
    rf.fit(X, y)
    rf_preds = rf.predict(X)
    assert rf_preds.shape == (100,)
    lr = CuMLWrapper("logistic_regression", max_iter=200)
    lr.fit(X, y)
    lr_preds = lr.predict(X)
    assert lr_preds.shape == (100,)
    svc = CuMLWrapper("svc", kernel="linear")
    svc.fit(X, y)
    svc_preds = svc.predict(X)
    assert svc_preds.shape == (100,)


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_wrapper_configuration_sanity():
    """Sanity test: Verify wrapper configuration handling."""
    X, y = generate_dummy_data()
    clf = CuMLWrapper("random_forest", n_estimators=5, fusion_head={"test": "config"})
    assert hasattr(clf, "wrapper_config")
    assert clf.wrapper_config.get("fusion_head") == {"test": "config"}
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_predict_proba_sanity():
    """Sanity test: Verify predict_proba functionality."""
    X, y = generate_dummy_data()
    rf = CuMLWrapper("random_forest", n_estimators=5)
    rf.fit(X, y)
    proba = rf.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_save_load_functionality_sanity():
    """Sanity test: Verify save and load functionality."""
    X, y = generate_dummy_data()
    clf = CuMLWrapper("random_forest", n_estimators=5)
    clf.fit(X, y)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        clf.save(tmp_path)
        assert tmp_path.exists()
        loaded_clf = CuMLWrapper.load("random_forest", tmp_path)
        loaded_preds = loaded_clf.predict(X)
        original_preds = clf.predict(X)
        np.testing.assert_array_equal(loaded_preds, original_preds)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_repr_functionality_unit():
    """Unit test: Verify string representation."""
    clf = CuMLWrapper("random_forest", n_estimators=5)
    repr_str = repr(clf)
    assert isinstance(repr_str, str)
    assert len(repr_str) > 0


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_invalid_model_path_unit():
    """Unit test: Verify error handling for invalid model paths."""
    with pytest.raises(ValueError, match="Invalid cuML model"):
        CuMLWrapper("invalid.model.path.Classifier")


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_invalid_module_unit():
    """Unit test: Verify error handling for invalid modules."""
    with pytest.raises(ValueError, match="Invalid cuML model"):
        CuMLWrapper("nonexistent.module.RandomForestClassifier")


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_predict_proba_not_supported_unit():
    """Unit test: Verify error handling when predict_proba is not supported."""
    X, y = generate_dummy_data()
    svc = CuMLWrapper("svc", kernel="linear")
    svc.fit(X, y)
    with pytest.raises(AttributeError, match="does not support predict_proba"):
        svc.predict_proba(X)


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_model_parameters_unit():
    """Unit test: Verify model parameter handling."""
    X, y = generate_dummy_data()
    clf1 = CuMLWrapper("random_forest", n_estimators=10, max_depth=3)
    clf1.fit(X, y)
    clf2 = CuMLWrapper("random_forest", n_estimators=5, max_depth=5)
    clf2.fit(X, y)
    preds1 = clf1.predict(X)
    preds2 = clf2.predict(X)


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_data_type_handling_unit():
    """Unit test: Verify data type handling."""
    X, y = generate_dummy_data()
    if CUML_AVAILABLE:
        X_float32 = X.astype(cp.float32)
        y_int32 = y.astype(cp.int32)
    else:
        X_float32 = X.astype(np.float32)
        y_int32 = y.astype(np.int32)
    clf = CuMLWrapper("random_forest", n_estimators=5)
    clf.fit(X_float32, y_int32)
    preds = clf.predict(X_float32)
    assert preds.shape == (100,)
    assert isinstance(preds, (np.ndarray, cp.ndarray))


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_empty_data_handling_unit():
    """Unit test: Verify handling of edge cases with empty data."""
    if CUML_AVAILABLE:
        X_single = cp.array([[1.0, 2.0, 3.0]])
        y_single = cp.array([0])
    else:
        X_single = np.array([[1.0, 2.0, 3.0]])
        y_single = np.array([0])
    clf = CuMLWrapper("random_forest", n_estimators=5)
    clf.fit(X_single, y_single)
    preds = clf.predict(X_single)
    assert preds.shape == (1,)
    assert isinstance(preds, (np.ndarray, cp.ndarray))


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_wrapper_config_isolation_unit():
    """Unit test: Verify wrapper config doesn't interfere with model params."""
    X, y = generate_dummy_data()
    clf = CuMLWrapper("random_forest", n_estimators=5, fusion_head={"test": "value"})
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)
    assert clf.wrapper_config.get("fusion_head") == {"test": "value"}


@pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
def test_cuml_specific_functionality_unit():
    """Unit test: Test cuML-specific functionality."""
    X, y = generate_dummy_data()
    clf = CuMLWrapper("random_forest", n_estimators=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    if CUML_AVAILABLE:
        assert isinstance(preds, cp.ndarray)
    else:
        assert isinstance(preds, np.ndarray)


def test_import_error_handling():
    """Test that appropriate error is raised when cuML is not available."""
    if not CUML_AVAILABLE:
        with pytest.raises(ImportError):
            CuMLWrapper("random_forest", n_estimators=5)
    else:
        pytest.skip("cuML is available")
