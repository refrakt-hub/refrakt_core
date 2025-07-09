"""
Common type aliases used across integration wrappers and registries.
"""

from typing import Any, TypeAlias

import numpy as np

NDArrayF: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]
NDArrayI: TypeAlias = np.ndarray[Any, np.dtype[np.int64]]
ClassifierOutput: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]
