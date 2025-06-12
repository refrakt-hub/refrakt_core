# test_mae.py
import pytest
import torch

from refrakt_core.models.mae import MAE


@pytest.fixture
def small_mae():
    return MAE(img_size=64, patch_size=8, encoder_depth=2, decoder_depth=2)

def test_mae_init(small_mae):
    assert small_mae.mask_ratio == 0.75
    assert small_mae.num_patches == (64 // 8) ** 2

def test_mae_forward(small_mae):
    x = torch.randn(2, 3, 64, 64)
    output = small_mae(x)
    
    assert "recon" in output
    assert "mask" in output
    assert "original_patches" in output
    
    assert output["recon"].shape == (2, 64, 192)
    assert output["original_patches"].shape == (2, 64, 192)
    assert output["mask"].shape == (2, 64)

def test_mae_patch_unpatch(small_mae):
    x = torch.randn(2, 3, 64, 64)
    patched = small_mae.patchify(x)
    unpatch = small_mae.unpatchify(patched)
    
    assert patched.shape == (2, 64, 192)
    assert unpatch.shape == (2, 3, 64, 64)