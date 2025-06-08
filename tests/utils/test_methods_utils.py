import torch
import pytest
from refrakt_core.utils.methods import (
    patchify, 
    positional_embeddings,
    random_masking,
    random_patch_masking
)

def test_patchify():
    images = torch.randn(2, 3, 32, 32)
    patches = patchify(images, n_patches=4)
    assert patches.shape == (2, 16, 3*8*8)

def test_positional_embeddings():
    embeddings = positional_embeddings(10, 64)
    assert embeddings.shape == (10, 64)
    assert not torch.allclose(embeddings[0], embeddings[1])

def test_random_masking():
    x = torch.randn(3, 10, 16)
    masked, mask, ids_restore, ids_keep = random_masking(x, 0.6)
    assert masked.shape == (3, 4, 16)
    assert mask.shape == (3, 10)
    assert ids_restore.shape == (3, 10)

def test_random_patch_masking():
    x = torch.randn(2, 3, 32, 32)
    masked = random_patch_masking(x, mask_ratio=0.5, patch_size=8)
    assert masked.shape == x.shape
    assert not torch.allclose(x, masked)