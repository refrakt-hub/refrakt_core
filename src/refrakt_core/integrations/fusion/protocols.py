# refrakt_core/integrations/fusion/protocols.py

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class FusionHead(Protocol):
    """
    Protocol for fusion head models (e.g., sklearn, cuml) used in FusionBlock.
    """

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit the fusion head model."""
        ...

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using the fusion head model."""
        ...

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities using the fusion head model."""
        ...
