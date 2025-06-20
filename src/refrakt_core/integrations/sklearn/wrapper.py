"""
Wrapper for dynamically loading and using scikit-learn models via a string-based registry.

This module allows you to specify sklearn models using simple string keys
or full class paths, dynamically instantiate them with parameters, and
use standard `fit`, `predict`, and `predict_proba` methods.

Example usage:
    >>> from refrakt_core.integrations.sklearn.wrapper import SklearnWrapper
    >>> clf = SklearnWrapper("random_forest", n_estimators=10, max_depth=5)
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    >>> _ = clf.fit(X, y)
    >>> preds = clf.predict(X)
    >>> isinstance(preds, (list, tuple, np.ndarray))
    True
"""

import importlib
from typing import Union, Protocol, NoReturn, cast, Type

from refrakt_core.integrations.sklearn.registry import load_sklearn_registry
from refrakt_core.integrations.types import NDArrayF, ClassifierOutput


class SklearnEstimator(Protocol):
    def fit(self, X: NDArrayF, y: NDArrayF) -> "SklearnEstimator": ...
    def predict(self, X: NDArrayF) -> ClassifierOutput: ...
    def predict_proba(self, X: NDArrayF) -> ClassifierOutput: ...


class SklearnWrapper:
    """
    A wrapper to instantiate and interact with sklearn models using either
    short registry keys or fully qualified class names.

    Attributes:
        model: The instantiated sklearn model.
    """

    def __init__(self, model: str, **params: Union[int, float, str, bool]):
        """
        Initialize the wrapper by loading a model from the registry or full import path.

        Args:
            model (str): Model key (e.g., "random_forest") or full class path.
            **params: Parameters for model instantiation.

        Raises:
            ValueError: If the model path is invalid.
        """
        registry = load_sklearn_registry()
        class_path = registry.get(model, model)

        module_path, class_name = class_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
            model_cls: Type[object] = getattr(module, class_name)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Invalid sklearn model '{model}': {e}")

        model_instance = model_cls(**params)
        self.model: SklearnEstimator = cast(SklearnEstimator, model_instance)

    def fit(self, X: NDArrayF, y: NDArrayF) -> SklearnEstimator:
        """
        Fit the wrapped model.

        Returns:
            SklearnEstimator: The fitted model instance.
        """
        return self.model.fit(X, y)

    def predict(self, X: NDArrayF) -> ClassifierOutput:
        """
        Run predictions using the model.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: NDArrayF) -> ClassifierOutput | NoReturn:
        """
        Predict class probabilities, if supported.

        Raises:
            AttributeError: If model lacks `predict_proba`.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError(
            f"{self.model.__class__.__name__} does not support predict_proba"
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the model.
        """
        return str(self.model)
