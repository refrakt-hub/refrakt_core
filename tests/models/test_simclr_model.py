# test_simclr.py
import pytest
import torch
import torch.nn.functional as F

from refrakt_core.models.simclr import SimCLRModel


@pytest.fixture
def small_simclr():
    return SimCLRModel(proj_dim=64)

def test_simclr_init(small_simclr):
    assert small_simclr.model_name == "simclr"
    assert isinstance(small_simclr.encoder, torch.nn.Module)
    assert isinstance(small_simclr.projector, torch.nn.Module)

def test_simclr_forward(small_simclr):
    x = torch.randn(2, 3, 224, 224)
    output = small_simclr(x)
    
    assert output.shape == (2, 64)
    # Verify normalization
    assert torch.allclose(torch.norm(output, dim=1), torch.ones(2), atol=1e-5)

def test_simclr_components(small_simclr):
    x = torch.randn(2, 3, 224, 224)
    
    # Test encode
    features = small_simclr.encode(x)
    assert features.shape == (2, 2048)
    
    # Test project
    projected = small_simclr.project(features)
    assert projected.shape == (2, 64)