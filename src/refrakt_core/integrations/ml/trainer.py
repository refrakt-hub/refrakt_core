from typing import Any, Dict, Optional
import numpy as np

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
        Xf = self.feature_pipeline.fit_transform(self.X_train)
        self.model.fit(Xf, self.y_train)
        metrics = {}
        if self.X_val is not None and self.y_val is not None:
            metrics = self.evaluate()
        return metrics

    def evaluate(self):
        Xf = self.feature_pipeline.transform(self.X_val)
        preds = self.model.predict(Xf)
        acc = (preds == self.y_val).mean()
        if self.artifact_dumper:
            self.artifact_dumper.log_scalar_dict({'ml_accuracy': acc}, step=0, prefix='val')
        return {'ml_accuracy': acc}

    def predict(self, X):
        Xf = self.feature_pipeline.transform(X)
        return self.model.predict(Xf) 