"""
Wrapper for dynamically loading and using cuML models via a string-based registry.

This module allows you to specify cuML models using simple string keys
or full class paths, dynamically instantiate them with parameters, and
use standard `fit`, `predict`, and `predict_proba` methods.

Example usage:
    >>> from refrakt_core.integrations.cuml.wrapper import CuMLWrapper
    >>> clf = CuMLWrapper("random_forest", n_estimators=10, max_depth=5)
    >>> import cupy as cp
    >>> X = cp.random.rand(50, 5)
    >>> y = cp.random.randint(0, 2, 50)
    >>> _ = clf.fit(X, y)
    >>> preds = clf.predict(X)
    >>> isinstance(preds, (list, tuple, cp.ndarray))
    True
"""

import importlib
from pathlib import Path
from typing import Any, Dict, NoReturn, Optional, Protocol, Type, Union, cast

import joblib
import numpy as np
from numpy.typing import NDArray
from refrakt_core.integrations.gpu.registry import load_cuml_registry
from refrakt_core.integrations.common_types import ClassifierOutput, NDArrayF


class CuMLEstimator(Protocol):
    def fit(self, X: NDArray, y: NDArray) -> "CuMLEstimator": ...
    def predict(self, X: NDArray) -> NDArray: ...
    def predict_proba(self, X: NDArray) -> NDArray: ...


class CuMLWrapper:
    """
    A wrapper to instantiate and interact with cuML models using either
    short registry keys or fully qualified class names.

    Attributes:
        model: The instantiated cuML model.
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
        registry = load_cuml_registry()
        class_path = registry.get(model, model)

        module_path, class_name = class_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
            model_cls: Type[object] = getattr(module, class_name)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Invalid cuML model '{model}': {e}")

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

        model_instance = model_cls(**model_params)
        self.model: CuMLEstimator = cast(CuMLEstimator, model_instance)

        # Store wrapper configuration
        self.wrapper_config = wrapper_params

    def fit(self, X: NDArrayF, y: NDArrayF) -> None:
        self.model.fit(X, y)

    def predict(self, X: NDArrayF) -> NDArray:
        return self.model.predict(X)

    def predict_proba(self, X: NDArrayF) -> NDArray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError(
            f"{self.model.__class__.__name__} does not support predict_proba"
        )

    def __repr__(self) -> str:
        return str(self.model)

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, model: str, path: Union[str, Path]) -> "CuMLWrapper":
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.wrapper_config = {}
        return instance
