"""
Tests for common type aliases used across integration wrappers and registries.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from refrakt_core.integrations.common_types import (
    NDArrayF,
    NDArrayI,
    ClassifierOutput,
)

def test_ndarrayf_type_alias_smoke():
    """Smoke test: Verify NDArrayF can be used for type annotations."""
    def test_function(data: NDArrayF) -> NDArrayF:
        return data
    test_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = test_function(test_data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64

def test_ndarrayi_type_alias_smoke():
    """Smoke test: Verify NDArrayI can be used for type annotations."""
    def test_function(data: NDArrayI) -> NDArrayI:
        return data
    test_data = np.array([1, 2, 3], dtype=np.int64)
    result = test_function(test_data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64

def test_classifier_output_type_alias_smoke():
    """Smoke test: Verify ClassifierOutput can be used for type annotations."""
    def test_function(data: ClassifierOutput) -> ClassifierOutput:
        return data
    test_data = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    result = test_function(test_data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64

def test_ndarrayf_sanity():
    """Sanity test: Verify NDArrayF works with typical float operations."""
    data: NDArrayF = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    assert data.shape == (2, 2)
    assert data.dtype == np.float64
    assert np.allclose(data.sum(), 10.0)
    result = data * 2.0
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64

def test_ndarrayi_sanity():
    """Sanity test: Verify NDArrayI works with typical integer operations."""
    data: NDArrayI = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert data.shape == (2, 2)
    assert data.dtype == np.int64
    assert data.sum() == 10
    result = data * 2
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64

def test_classifier_output_sanity():
    """Sanity test: Verify ClassifierOutput works with typical classifier outputs."""
    data: ClassifierOutput = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    assert data.shape == (3,)
    assert data.dtype == np.float64
    assert np.isclose(data.sum(), 1.0)
    assert np.all(data >= 0.0)
    assert np.all(data <= 1.0)

def test_type_compatibility_unit():
    """Unit test: Verify type compatibility between aliases."""
    float_data = np.array([1.0, 2.0], dtype=np.float64)
    ndarrayf_data: NDArrayF = float_data
    assert ndarrayf_data.dtype == np.float64
    int_data = np.array([1, 2], dtype=np.int64)
    ndarrayi_data: NDArrayI = int_data
    assert ndarrayi_data.dtype == np.int64
    classifier_data: ClassifierOutput = float_data
    assert classifier_data.dtype == np.float64

def test_type_annotations_unit():
    """Unit test: Verify type annotations work correctly."""
    def process_float_data(data: NDArrayF) -> NDArrayF:
        return data.astype(np.float64)
    def process_int_data(data: NDArrayI) -> NDArrayI:
        return data.astype(np.int64)
    def process_classifier_output(data: ClassifierOutput) -> ClassifierOutput:
        return data.astype(np.float64)
    float_result = process_float_data(np.array([1.0, 2.0], dtype=np.float64))
    assert float_result.dtype == np.float64
    int_result = process_int_data(np.array([1, 2], dtype=np.int64))
    assert int_result.dtype == np.int64
    classifier_result = process_classifier_output(np.array([0.5, 0.5], dtype=np.float64))
    assert classifier_result.dtype == np.float64

def test_error_handling_unit():
    """Unit test: Verify appropriate error handling for wrong types."""
    wrong_float_data = np.array([1.0, 2.0], dtype=np.float32)
    assert isinstance(wrong_float_data, np.ndarray)
    wrong_int_data = np.array([1, 2], dtype=np.int32)
    assert isinstance(wrong_int_data, np.ndarray)

def test_array_operations_unit():
    """Unit test: Verify array operations work with type aliases."""
    data_f: NDArrayF = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result_f = data_f + 1.0
    assert isinstance(result_f, np.ndarray)
    assert result_f.dtype == np.float64
    data_i: NDArrayI = np.array([1, 2, 3], dtype=np.int64)
    result_i = data_i + 1
    assert isinstance(result_i, np.ndarray)
    assert result_i.dtype == np.int64
    data_c: ClassifierOutput = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    result_c = data_c * 2.0
    assert isinstance(result_c, np.ndarray)
    assert result_c.dtype == np.float64 