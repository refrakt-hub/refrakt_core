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
from pathlib import Path
from typing import Any, Dict, NoReturn, Optional, Protocol, Type, Union, cast

import joblib
import numpy as np
from numpy.typing import NDArray
from refrakt_core.integrations.cpu.registry import load_sklearn_registry
from refrakt_core.integrations.common_types import ClassifierOutput, NDArrayF


class SklearnEstimator(Protocol):
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "SklearnEstimator": ...
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


class SklearnWrapper:
    """
    A wrapper to instantiate and interact with sklearn models using either
    short registry keys or fully qualified class names.

    Attributes:
        model: The instantiated sklearn model.
    """

    def __init__(self, model: str, **params: Dict[str, Any]):
        """
        Initialize the wrapper by loading a model from the registry or full import path.

        Args:
            model (str): Model key (e.g., "random_forest") or full class path.
            **params: Parameters dictionary containing:
                - Model parameters for instantiation
                - Special keys like 'fusion_head' for wrapper configuration

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

        # Extract wrapper-specific parameters
        wrapper_params = {}
        model_params = dict(params)  # Make a copy to modify

        # Handle special parameters
        if "fusion_head" in model_params:
            fusion_head_val = model_params.pop("fusion_head")
            if fusion_head_val is None:
                wrapper_params["fusion_head"] = {}
            else:
                wrapper_params["fusion_head"] = fusion_head_val
        print(f"DEBUG (SklearnWrapper): wrapper_params={wrapper_params}, model_params={model_params}")

        model_instance = model_cls(**model_params)
        self.model: SklearnEstimator = cast(SklearnEstimator, model_instance)

        # Store wrapper configuration
        self.wrapper_config = wrapper_params

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Fit the wrapped model.
        """
        self.model.fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Run predictions using the model.

        Returns:
            NDArray[np.float64]: Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
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

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, model: str, path: Union[str, Path]) -> "SklearnWrapper":
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.wrapper_config = {}
        return instance
