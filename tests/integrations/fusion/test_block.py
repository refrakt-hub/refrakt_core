"""
Comprehensive tests for fusion block module.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock

from refrakt_core.integrations.fusion.block import FusionBlock
from refrakt_core.integrations.fusion.builder import build_fusion_head
from refrakt_core.integrations.cpu.wrapper import SklearnWrapper
from refrakt_core.schema.model_output import ModelOutput


class DummyBackbone(nn.Module):
    """Dummy backbone for testing."""
    
    def __init__(self, feature_dim: int = 10):
        super().__init__()
        self.fc = nn.Linear(20, feature_dim)
        
    def forward(self, x):
        # Handle dict input for MSN-style blocks
        if isinstance(x, dict):
            x = x["anchor"]
        embeddings = self.fc(x)
        return ModelOutput(embeddings=embeddings)


class AutoencoderBackbone(nn.Module):
    """Dummy generative backbone for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 10)
        
    def forward(self, x):
        return ModelOutput(embeddings=self.fc(x))


def create_dummy_data(n_samples: int = 100, n_features: int = 20):
    """Create dummy data for testing."""
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))
    return X, y


def test_fusion_block_initialization_smoke():
    """Smoke test: FusionBlock initializes correctly."""
    backbone = DummyBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    
    assert fusion_block.backbone == backbone
    assert hasattr(fusion_block, 'fusion_head')
    assert fusion_block._trained == False
    assert fusion_block.wrapper_config["wrapper_type"] == "fusion"


def test_fusion_block_fit_smoke():
    """Smoke test: FusionBlock can fit and make predictions."""
    backbone = DummyBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    X, y = create_dummy_data()
    
    # Fit the fusion block
    fusion_block.fit(X, y)
    assert fusion_block._trained == True
    
    # Make predictions
    fusion_block.eval()
    with torch.no_grad():
        output = fusion_block(X)
    
    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.extra is not None
    assert "fusion_preds" in output.extra


def test_fusion_block_forward_training_mode_smoke():
    """Smoke test: FusionBlock forward in training mode."""
    backbone = DummyBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    X, y = create_dummy_data()
    
    # In training mode, should return embeddings
    fusion_block.train()
    output = fusion_block(X)
    
    assert isinstance(output, ModelOutput)
    assert output.embeddings is not None


def test_fusion_block_parameters_sanity():
    """Sanity test: Verify parameters are accessible."""
    backbone = DummyBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    
    # Check that parameters are accessible
    params = list(fusion_block.parameters())
    assert len(params) > 0
    
    # Check that backbone parameters are included
    backbone_params = list(backbone.parameters())
    assert len(params) == len(backbone_params)


def test_fusion_block_feature_extraction_sanity():
    """Sanity test: Verify feature extraction works correctly."""
    backbone = DummyBackbone(feature_dim=5)
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    X, y = create_dummy_data()
    
    # Extract features
    feats, output = fusion_block._extract_features(X)
    
    assert isinstance(feats, np.ndarray)
    assert feats.shape[1] == 5  # Feature dimension
    assert isinstance(output, ModelOutput)
    assert output.embeddings is not None


def test_fusion_block_dict_input_sanity():
    """Sanity test: Verify handling of dict input (for MSN)."""
    backbone = DummyBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    X, y = create_dummy_data()
    
    # Create dict input
    dict_input = {"anchor": X}
    
    # Fit first
    fusion_block.fit(X, y)
    fusion_block.eval()
    
    # Test with dict input
    with torch.no_grad():
        output = fusion_block(dict_input)
    
    assert isinstance(output, ModelOutput)
    assert output.logits is not None


def test_fusion_block_predict_proba_sanity():
    """Sanity test: Verify predict_proba functionality."""
    backbone = DummyBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    X, y = create_dummy_data()
    
    # Fit the fusion block
    fusion_block.fit(X, y)
    
    # Test predict_proba
    proba = fusion_block.predict_proba(X)
    assert proba is not None
    assert isinstance(proba, np.ndarray)


def test_fusion_block_forward_for_graph_sanity():
    """Sanity test: Verify forward_for_graph functionality."""
    backbone = DummyBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    X, y = create_dummy_data()
    
    # Test forward_for_graph
    output = fusion_block.forward_for_graph(X)
    assert isinstance(output, torch.Tensor)


def test_generative_model_error_unit():
    """Unit test: Verify error for generative models."""
    backbone = AutoencoderBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    with pytest.raises(NotImplementedError, match="Fusion is not yet supported for generative models"):
        FusionBlock(backbone, fusion_cfg)


def test_no_embeddings_error_unit():
    """Unit test: Verify error when backbone doesn't return embeddings."""
    class NoEmbeddingsBackbone(nn.Module):
        def forward(self, x):
            return ModelOutput(logits=torch.randn(x.shape[0], 2))
    
    backbone = NoEmbeddingsBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    X, y = create_dummy_data()
    
    with pytest.raises(ValueError, match="Backbone did not return embeddings in ModelOutput"):
        fusion_block._extract_features(X)


def test_teacher_update_unit():
    """Unit test: Verify teacher update delegation."""
    class TeacherBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(20, 10)
            
        def forward(self, x):
            return ModelOutput(embeddings=self.fc(x))
            
        def update_teacher(self, *args, **kwargs):
            return "teacher_updated"
    
    backbone = TeacherBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    result = fusion_block.update_teacher()
    assert result == "teacher_updated"


def test_no_teacher_update_error_unit():
    """Unit test: Verify error when backbone doesn't support teacher update."""
    backbone = DummyBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    
    with pytest.raises(AttributeError, match="Backbone does not support update_teacher"):
        fusion_block.update_teacher()


def test_predict_proba_not_trained_unit():
    """Unit test: Verify predict_proba returns None when not trained."""
    backbone = DummyBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    X, y = create_dummy_data()
    
    # Should return None when not trained
    proba = fusion_block.predict_proba(X)
    assert proba is None


def test_forward_with_teacher_parameter_unit():
    """Unit test: Verify forward with teacher parameter."""
    class TeacherBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(20, 10)
            
        def forward(self, x, teacher=False, **kwargs):
            return ModelOutput(embeddings=self.fc(x))
    
    backbone = TeacherBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    X, y = create_dummy_data()
    
    # Test with teacher parameter
    output = fusion_block(X, teacher=True)
    assert isinstance(output, ModelOutput)


def test_forward_with_none_embeddings_unit():
    """Unit test: Verify forward with None embeddings."""
    class NoneEmbeddingsBackbone(nn.Module):
        def forward(self, x):
            return ModelOutput(embeddings=None)
    
    backbone = NoneEmbeddingsBackbone()
    fusion_cfg = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 5}
    }
    
    fusion_block = FusionBlock(backbone, fusion_cfg)
    X, y = create_dummy_data()
    
    # Should handle None embeddings gracefully
    output = fusion_block(X)
    assert isinstance(output, ModelOutput) 