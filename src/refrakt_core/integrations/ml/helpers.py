"""
Helper functions for ML trainer decomposition.
"""

import numpy as np
from typing import Any, Dict


def prepare_training_data(feature_pipeline, X_train, y_train):
    """
    Prepare training data using the feature pipeline.
    
    Args:
        feature_pipeline: The feature transformation pipeline
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Transformed training features
    """
    return feature_pipeline.fit_transform(X_train)


def prepare_evaluation_data(feature_pipeline, X_val):
    """
    Prepare evaluation data using the feature pipeline.
    
    Args:
        feature_pipeline: The feature transformation pipeline
        X_val: Validation features
        
    Returns:
        Transformed validation features
    """
    return feature_pipeline.transform(X_val)


def calculate_accuracy(predictions, y_true):
    """
    Calculate accuracy from predictions and true labels.
    
    Args:
        predictions: Model predictions
        y_true: True labels
        
    Returns:
        Accuracy score
    """
    return (predictions == y_true).mean()


def log_metrics(artifact_dumper, metrics, step=0, prefix='val'):
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