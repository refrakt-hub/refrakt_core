from typing import Any, Dict, Tuple, Optional
from refrakt_core.integrations.ml.feature_engineering import build_feature_pipeline
from refrakt_core.integrations.ml.wrapper import build_ml_model
import numpy as np


def build_ml_pipeline(cfg: Dict[str, Any], X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
    """
    Build the feature pipeline and ML model from config, and return them with data splits.
    cfg should have 'feature_engineering' (list of steps) and 'model' (dict).
    """
    feature_steps = cfg.get('feature_engineering', [])
    feature_pipeline = build_feature_pipeline(feature_steps) if feature_steps else None
    ml_model = build_ml_model(cfg['model'])
    return feature_pipeline, ml_model, X, y, X_val, y_val 