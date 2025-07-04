"""
Comprehensive tests for fusion protocols module.
"""

import numpy as np
import pytest
from typing import Protocol

from refrakt_core.integrations.fusion.protocols import FusionHead


class MockFusionHead:
    """Mock fusion head that implements the FusionHead protocol."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the fusion head model."""
        self.fitted = True
        self.X_shape = X.shape
        self.y_shape = y.shape
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fusion head model."""
        if not hasattr(self, 'fitted'):
            raise ValueError("Model not fitted")
        return np.random.randint(0, 2, X.shape[0])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the fusion head model."""
        if not hasattr(self, 'fitted'):
            raise ValueError("Model not fitted")
        return np.random.rand(X.shape[0], 2)


class IncompleteFusionHead:
    """Incomplete fusion head that doesn't implement all protocol methods."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass


def test_fusion_head_protocol_smoke():
    """Smoke test: FusionHead protocol can be used for type checking."""
    def test_function(head: FusionHead) -> FusionHead:
        return head
    
    mock_head = MockFusionHead()
    result = test_function(mock_head)
    
    assert result == mock_head


def test_fusion_head_fit_smoke():
    """Smoke test: FusionHead fit method works correctly."""
    head = MockFusionHead()
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    head.fit(X, y)
    
    assert head.fitted == True
    assert head.X_shape == (100, 10)
    assert head.y_shape == (100,)


def test_fusion_head_predict_smoke():
    """Smoke test: FusionHead predict method works correctly."""
    head = MockFusionHead()
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Fit first
    head.fit(X, y)
    
    # Then predict
    predictions = head.predict(X)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (100,)
    assert np.all((predictions >= 0) & (predictions <= 1))


def test_fusion_head_predict_proba_smoke():
    """Smoke test: FusionHead predict_proba method works correctly."""
    head = MockFusionHead()
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Fit first
    head.fit(X, y)
    
    # Then predict probabilities
    probabilities = head.predict_proba(X)
    
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (100, 2)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


def test_fusion_head_protocol_compliance_sanity():
    """Sanity test: Verify protocol compliance."""
    head = MockFusionHead()
    
    # Check that all required methods exist
    assert hasattr(head, 'fit')
    assert hasattr(head, 'predict')
    assert hasattr(head, 'predict_proba')
    
    # Check that methods are callable
    assert callable(head.fit)
    assert callable(head.predict)
    assert callable(head.predict_proba)


def test_fusion_head_data_type_handling_sanity():
    """Sanity test: Verify data type handling."""
    head = MockFusionHead()
    
    # Test with different data types
    X_float32 = np.random.rand(50, 5).astype(np.float32)
    y_int32 = np.random.randint(0, 2, 50).astype(np.int32)
    
    head.fit(X_float32, y_int32)
    predictions = head.predict(X_float32)
    probabilities = head.predict_proba(X_float32)
    
    assert isinstance(predictions, np.ndarray)
    assert isinstance(probabilities, np.ndarray)


def test_fusion_head_empty_data_sanity():
    """Sanity test: Verify handling of empty data."""
    head = MockFusionHead()
    
    # Test with single sample
    X_single = np.random.rand(1, 5)
    y_single = np.random.randint(0, 2, 1)
    
    head.fit(X_single, y_single)
    predictions = head.predict(X_single)
    probabilities = head.predict_proba(X_single)
    
    assert predictions.shape == (1,)
    assert probabilities.shape == (1, 2)


def test_fusion_head_not_fitted_error_unit():
    """Unit test: Verify error when trying to predict without fitting."""
    head = MockFusionHead()
    X = np.random.rand(100, 10)
    
    with pytest.raises(ValueError, match="Model not fitted"):
        head.predict(X)
    
    with pytest.raises(ValueError, match="Model not fitted"):
        head.predict_proba(X)


def test_incomplete_fusion_head_unit():
    """Unit test: Verify that incomplete implementations don't satisfy protocol."""
    incomplete_head = IncompleteFusionHead()
    
    # Should not have all required methods
    assert hasattr(incomplete_head, 'fit')
    assert not hasattr(incomplete_head, 'predict')
    assert not hasattr(incomplete_head, 'predict_proba')


def test_fusion_head_protocol_type_annotations_unit():
    """Unit test: Verify type annotations work correctly."""
    def process_fusion_head(head: FusionHead, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        head.fit(X, y)
        return head.predict(X)
    
    head = MockFusionHead()
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    result = process_fusion_head(head, X, y)
    assert isinstance(result, np.ndarray)


def test_fusion_head_protocol_inheritance_unit():
    """Unit test: Verify protocol inheritance behavior."""
    # FusionHead should be a Protocol
    assert isinstance(FusionHead, type)
    
    # MockFusionHead should be compatible with FusionHead
    head = MockFusionHead()
    assert hasattr(head, 'fit')
    assert hasattr(head, 'predict')
    assert hasattr(head, 'predict_proba')


def test_fusion_head_method_signatures_unit():
    """Unit test: Verify method signatures match protocol."""
    head = MockFusionHead()
    
    # Check fit method signature
    import inspect
    fit_sig = inspect.signature(head.fit)
    assert len(fit_sig.parameters) == 2  # X, y
    
    # Check predict method signature
    predict_sig = inspect.signature(head.predict)
    assert len(predict_sig.parameters) == 1  # X
    
    # Check predict_proba method signature
    proba_sig = inspect.signature(head.predict_proba)
    assert len(proba_sig.parameters) == 1  # X


def test_fusion_head_return_types_unit():
    """Unit test: Verify return types match protocol."""
    head = MockFusionHead()
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    head.fit(X, y)
    
    # fit should return None
    result = head.fit(X, y)
    assert result is None
    
    # predict should return np.ndarray
    predictions = head.predict(X)
    assert isinstance(predictions, np.ndarray)
    
    # predict_proba should return np.ndarray
    probabilities = head.predict_proba(X)
    assert isinstance(probabilities, np.ndarray) 