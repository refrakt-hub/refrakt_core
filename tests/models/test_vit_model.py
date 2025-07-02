# test_vit.py
import pytest
import torch

from refrakt_core.models.vit import VisionTransformer


@pytest.fixture
def small_vit():
    return VisionTransformer(
        image_size=64, patch_size=8, num_classes=10, dim=128, depth=2, heads=2
    )


def test_vit_init(small_vit):
    assert small_vit.model_name == "vit_classifier"
    assert small_vit.n_patches == 8  # 64//8
    assert len(small_vit.blocks) == 2


def test_vit_forward(small_vit):
    x = torch.randn(2, 3, 64, 64)
    output = small_vit(x)
    assert output.shape == (2, 10)


def test_vit_features(small_vit):
    x = torch.randn(2, 3, 64, 64)
    features = small_vit.features(x)
    assert features.shape == (2, 128)  # CLS token dimension


def test_vit_patchify(small_vit):
    x = torch.randn(1, 3, 64, 64)
    patches = small_vit.forward_features(x)
    assert patches.shape == (1, 128)  # (n_patches^2 + 1, dim)
