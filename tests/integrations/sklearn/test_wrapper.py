import numpy as np
from sklearn.datasets import make_classification

from refrakt_core.integrations.sklearn.wrapper import SklearnWrapper


def generate_dummy_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data for testing.

    Returns:
        Tuple of features and labels (X, y).
    """
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y


def test_model_instantiates_from_yaml_registry() -> None:
    """
    Test that a model loads from the YAML registry and makes predictions.
    """
    X, y = generate_dummy_data()
    clf = SklearnWrapper("random_forest", n_estimators=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)


def test_model_instantiates_from_full_path() -> None:
    """
    Test that a model loads from a full class path and makes predictions.
    """
    X, y = generate_dummy_data()
    clf = SklearnWrapper("sklearn.linear_model.LogisticRegression", max_iter=200)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)
