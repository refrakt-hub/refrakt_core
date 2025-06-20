# refrakt_core/integrations/fusion/protocols.py

from typing import Protocol
import numpy as np

class FusionHead(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

