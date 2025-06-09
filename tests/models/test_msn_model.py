# test_msn.py
# test_msn.py
import pytest
import torch

from refrakt_core.models.msn import MSNModel


@pytest.fixture
def small_msn():
    return MSNModel(
        encoder_name="resnet18",
        projector_dim=128,
        num_prototypes=100,
        pretrained=False
    )

def test_msn_init(small_msn):
    # Instead of checking model_name, verify the model has the required components
    assert hasattr(small_msn, 'encoder')
    assert hasattr(small_msn, 'target_encoder')
    assert hasattr(small_msn, 'projector')
    assert hasattr(small_msn, 'target_projector')
    assert hasattr(small_msn, 'prototypes')
    
    # Verify target encoder is frozen
    for param in small_msn.target_encoder.parameters():
        assert not param.requires_grad

def test_msn_forward(small_msn):
    x_anchor = torch.randn(2, 3, 224, 224)
    x_target = torch.randn(2, 3, 224, 224)
    
    z_anchor, z_target, prototypes = small_msn(x_anchor, x_target)
    
    assert z_anchor.shape == (2, 128)
    assert z_target.shape == (2, 128)
    assert prototypes.shape == (100, 128)
    
    # Verify normalization
    assert torch.allclose(torch.norm(z_anchor, dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(torch.norm(z_target, dim=1), torch.ones(2), atol=1e-5)

def test_msn_forward(small_msn):
    x_anchor = torch.randn(2, 3, 224, 224)
    x_target = torch.randn(2, 3, 224, 224)
    
    z_anchor, z_target, prototypes = small_msn(x_anchor, x_target)
    
    assert z_anchor.shape == (2, 128)
    assert z_target.shape == (2, 128)
    assert prototypes.shape == (100, 128)
    
    # Verify normalization
    assert torch.allclose(torch.norm(z_anchor, dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(torch.norm(z_target, dim=1), torch.ones(2), atol=1e-5)