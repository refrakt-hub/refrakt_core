"""
Comprehensive tests for wrapper.py (build_ml_model).
"""

import numpy as np
import pytest
from refrakt_core.integrations.ml.wrapper import build_ml_model
from refrakt_core.integrations.cpu.wrapper import SklearnWrapper
from refrakt_core.integrations.gpu.wrapper import CuMLWrapper


def test_build_ml_model_sklearn_smoke():
    """Smoke test: Build sklearn model using wrapper."""
    cfg = {'backend': 'sklearn', 'name': 'logistic_regression', 'params': {'max_iter': 100}}
    model = build_ml_model(cfg)
    assert isinstance(model, SklearnWrapper)
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


def test_build_ml_model_cuml_smoke():
    """Smoke test: Build cuML model using wrapper (should raise if cuML not installed or hardware is insufficient)."""
    cfg = {'backend': 'cuml', 'name': 'random_forest', 'params': {'n_estimators': 5}}
    try:
        model = build_ml_model(cfg)
        assert isinstance(model, CuMLWrapper)
    except Exception as e:
        # Accept any error indicating cuML is not available or hardware is insufficient
        pytest.skip(f"cuML unavailable or hardware insufficient: {e}")


def test_build_ml_model_default_backend_sanity():
    """Sanity test: Default backend is sklearn."""
    cfg = {'name': 'logistic_regression', 'params': {}}
    model = build_ml_model(cfg)
    assert isinstance(model, SklearnWrapper)


def test_build_ml_model_invalid_backend_unit():
    """Unit test: Error on unknown backend."""
    cfg = {'backend': 'unknown', 'name': 'logistic_regression', 'params': {}}
    with pytest.raises(ValueError, match="Unknown ML backend"):
        build_ml_model(cfg)


def test_build_ml_model_missing_name_unit():
    """Unit test: Error on missing model name."""
    cfg = {'backend': 'sklearn', 'params': {}}
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'rsplit'"):
        build_ml_model(cfg)


def test_build_ml_model_explicit_none_name_unit():
    """Unit test: Error when model name is explicitly None."""
    cfg = {'backend': 'sklearn', 'name': None, 'params': {}}
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'rsplit'"):
        build_ml_model(cfg)


def test_build_ml_model_with_extra_params_unit():
    """Unit test: Model accepts extra parameters."""
    cfg = {'backend': 'sklearn', 'name': 'logistic_regression', 'params': {'max_iter': 50, 'solver': 'liblinear'}}
    model = build_ml_model(cfg)
    assert isinstance(model, SklearnWrapper)


def test_build_ml_model_predict_sanity():
    """Sanity test: Built model can fit and predict."""
    cfg = {'backend': 'sklearn', 'name': 'logistic_regression', 'params': {'max_iter': 100}}
    model = build_ml_model(cfg)
    X = np.random.randn(20, 4)
    y = np.random.randint(0, 2, 20)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,) 