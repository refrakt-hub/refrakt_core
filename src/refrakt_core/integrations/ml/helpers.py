"""
Helper functions for ML trainer decomposition.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.pipeline import Pipeline  # type: ignore


def prepare_training_data(
    feature_pipeline: Pipeline,
    X_train: np.ndarray[Any, np.dtype[Any]],
    y_train: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Prepare training data using the feature pipeline.

    Args:
        feature_pipeline: The feature transformation pipeline
        X_train: Training features
        y_train: Training labels

    Returns:
        Transformed training features
    """
    return feature_pipeline.fit_transform(X_train)  # type: ignore


def prepare_evaluation_data(
    feature_pipeline: Pipeline, X_val: np.ndarray[Any, np.dtype[Any]]
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Prepare evaluation data using the feature pipeline.

    Args:
        feature_pipeline: The feature transformation pipeline
        X_val: Validation features

    Returns:
        Transformed validation features
    """
    return feature_pipeline.transform(X_val)  # type: ignore


def calculate_accuracy(
    predictions: np.ndarray[Any, np.dtype[Any]], y_true: np.ndarray[Any, np.dtype[Any]]
) -> float:
    """
    Calculate accuracy from predictions and true labels.

    Args:
        predictions: Model predictions
        y_true: True labels

    Returns:
        Accuracy score
    """
    return float((predictions == y_true).mean())


def log_metrics(
    artifact_dumper: Optional[Any],
    metrics: Dict[str, float],
    step: int = 0,
    prefix: str = "val",
) -> None:
    """
    Log metrics using the artifact dumper if available.

    Args:
        artifact_dumper: Artifact dumper instance
        metrics: Dictionary of metrics to log
        step: Step number for logging
        prefix: Prefix for metric names
    """
    if artifact_dumper:
        artifact_dumper.log_scalar_dict(metrics, step=step, prefix=prefix)
