from typing import Any, Dict, Optional

import numpy as np
from sklearn.pipeline import Pipeline  # type: ignore

from refrakt_core.integrations.ml.helpers import (
    calculate_accuracy,
    log_metrics,
    prepare_evaluation_data,
    prepare_training_data,
)


class MLTrainer:
    def __init__(
        self,
        feature_pipeline: Pipeline,
        model: Any,
        X_train: np.ndarray[Any, np.dtype[Any]],
        y_train: np.ndarray[Any, np.dtype[Any]],
        X_val: Optional[np.ndarray[Any, np.dtype[Any]]] = None,
        y_val: Optional[np.ndarray[Any, np.dtype[Any]]] = None,
        artifact_dumper: Optional[Any] = None,
    ) -> None:
        self.feature_pipeline = feature_pipeline
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.artifact_dumper = artifact_dumper

    def train(self) -> Dict[str, float]:
        Xf = prepare_training_data(self.feature_pipeline, self.X_train, self.y_train)
        self.model.fit(Xf, self.y_train)
        metrics = {}
        if self.X_val is not None and self.y_val is not None:
            metrics = self.evaluate()
        return metrics

    def evaluate(self) -> Dict[str, float]:
        if self.X_val is None or self.y_val is None:
            raise ValueError("X_val and y_val must be provided for evaluation")
        Xf = prepare_evaluation_data(self.feature_pipeline, self.X_val)
        preds = self.model.predict(Xf)
        acc = calculate_accuracy(preds, self.y_val)
        log_metrics(self.artifact_dumper, {"ml_accuracy": acc}, step=0, prefix="val")
        return {"ml_accuracy": acc}

    def predict(
        self, X: np.ndarray[Any, np.dtype[Any]]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        Xf = prepare_evaluation_data(self.feature_pipeline, X)
        return self.model.predict(Xf)  # type: ignore
