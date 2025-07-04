"""
Comprehensive tests for MLTrainer in trainer.py.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from refrakt_core.integrations.ml.trainer import MLTrainer


def generate_dummy_data():
    X = np.random.randn(40, 5)
    y = np.random.randint(0, 2, 40)
    return X, y


def test_ml_trainer_smoke():
    """Smoke test: MLTrainer can train and predict."""
    X, y = generate_dummy_data()
    feature_pipeline = StandardScaler()
    model = LogisticRegression(max_iter=100)
    trainer = MLTrainer(feature_pipeline, model, X, y)
    metrics = trainer.train()
    assert isinstance(metrics, dict)
    preds = trainer.predict(X)
    assert preds.shape == (X.shape[0],)


def test_ml_trainer_with_validation_sanity():
    """Sanity test: MLTrainer evaluates on validation data."""
    X, y = generate_dummy_data()
    X_val, y_val = generate_dummy_data()
    feature_pipeline = StandardScaler()
    model = LogisticRegression(max_iter=100)
    trainer = MLTrainer(feature_pipeline, model, X, y, X_val, y_val)
    metrics = trainer.train()
    assert 'ml_accuracy' in metrics
    assert 0.0 <= metrics['ml_accuracy'] <= 1.0


def test_ml_trainer_predict_sanity():
    """Sanity test: MLTrainer predict returns correct shape."""
    X, y = generate_dummy_data()
    feature_pipeline = StandardScaler()
    model = LogisticRegression(max_iter=100)
    trainer = MLTrainer(feature_pipeline, model, X, y)
    trainer.train()
    preds = trainer.predict(X)
    assert preds.shape == (X.shape[0],)


def test_ml_trainer_evaluate_unit():
    """Unit test: MLTrainer evaluate returns accuracy."""
    X, y = generate_dummy_data()
    X_val, y_val = generate_dummy_data()
    feature_pipeline = StandardScaler()
    model = LogisticRegression(max_iter=100)
    trainer = MLTrainer(feature_pipeline, model, X, y, X_val, y_val)
    trainer.train()
    metrics = trainer.evaluate()
    assert 'ml_accuracy' in metrics
    assert 0.0 <= metrics['ml_accuracy'] <= 1.0


def test_ml_trainer_artifact_dumper_unit():
    """Unit test: MLTrainer calls artifact_dumper if provided."""
    X, y = generate_dummy_data()
    X_val, y_val = generate_dummy_data()
    feature_pipeline = StandardScaler()
    model = LogisticRegression(max_iter=100)
    class DummyDumper:
        def __init__(self):
            self.logged = False
        def log_scalar_dict(self, d, step, prefix):
            self.logged = True
    dumper = DummyDumper()
    trainer = MLTrainer(feature_pipeline, model, X, y, X_val, y_val, artifact_dumper=dumper)
    trainer.train()
    assert dumper.logged


def test_ml_trainer_predict_before_train_unit():
    """Unit test: Predict before train raises NotFittedError."""
    from sklearn.exceptions import NotFittedError
    X, y = generate_dummy_data()
    feature_pipeline = StandardScaler()
    model = LogisticRegression(max_iter=100)
    trainer = MLTrainer(feature_pipeline, model, X, y)
    with pytest.raises(NotFittedError):
        trainer.predict(X)


def test_ml_trainer_evaluate_without_validation_unit():
    """Unit test: Evaluate without validation data raises AttributeError."""
    X, y = generate_dummy_data()
    feature_pipeline = StandardScaler()
    model = LogisticRegression(max_iter=100)
    trainer = MLTrainer(feature_pipeline, model, X, y)
    with pytest.raises(AttributeError):
        trainer.evaluate() 