from typing import Any, Dict, Optional
import numpy as np

from refrakt_core.integrations.ml.helpers import (
    prepare_training_data,
    prepare_evaluation_data,
    calculate_accuracy,
    log_metrics
)

class MLTrainer:
    def __init__(self, feature_pipeline, model, X_train, y_train, X_val=None, y_val=None, artifact_dumper=None):
        self.feature_pipeline = feature_pipeline
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.artifact_dumper = artifact_dumper

    def train(self):
        Xf = prepare_training_data(self.feature_pipeline, self.X_train, self.y_train)
        self.model.fit(Xf, self.y_train)
        metrics = {}
        if self.X_val is not None and self.y_val is not None:
            metrics = self.evaluate()
        return metrics

    def evaluate(self):
        Xf = prepare_evaluation_data(self.feature_pipeline, self.X_val)
        preds = self.model.predict(Xf)
        acc = calculate_accuracy(preds, self.y_val)
        log_metrics(self.artifact_dumper, {'ml_accuracy': acc}, step=0, prefix='val')
        return {'ml_accuracy': acc}

    def predict(self, X):
        Xf = prepare_evaluation_data(self.feature_pipeline, X)
        return self.model.predict(Xf) 