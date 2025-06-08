import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from refrakt_core.losses.perceptual import PerceptualLoss


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def loss_fn(device):
    return PerceptualLoss(device=device)


@pytest.fixture
def dummy_inputs(device):
    sr = torch.randn(1, 3, 224, 224, device=device)
    hr = torch.randn(1, 3, 224, 224, device=device)
    return sr, hr


def test_loss_output_is_scalar(loss_fn, dummy_inputs):
    sr, hr = dummy_inputs
    loss = loss_fn(sr, hr)
    assert loss.dim() == 0, "Loss should be a scalar tensor"


def test_identical_inputs_returns_zero(loss_fn, device):
    tensor = torch.randn(1, 3, 224, 224, device=device)
    loss = loss_fn(tensor, tensor)
    assert torch.isclose(
        loss, torch.tensor(0.0, device=device), atol=1e-5
    ), f"Expected loss to be ~0.0 but got {loss.item()}"


def test_backward_pass(loss_fn, dummy_inputs):
    sr, hr = dummy_inputs
    sr.requires_grad_()
    loss = loss_fn(sr, hr)
    loss.backward()
    assert sr.grad is not None, "Gradients were not computed for the input"


def test_shape_mismatch_raises_error(loss_fn, device):
    sr = torch.randn(1, 3, 224, 224, device=device)
    hr = torch.randn(1, 3, 128, 128, device=device)  # Wrong shape
    with pytest.raises(ValueError, match="Feature shape mismatch"):
        loss_fn(sr, hr)


def test_get_config(loss_fn):
    config = loss_fn.get_config()
    assert config["backbone"] == "vgg19"
    assert config["layers_used"] == "features[:36]"


def test_cpu_compatibility():
    loss_fn_cpu = PerceptualLoss(device="cpu")
    sr = torch.randn(1, 3, 224, 224)
    hr = torch.randn(1, 3, 224, 224)
    loss = loss_fn_cpu(sr, hr)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_batched_inputs(loss_fn, device):
    sr = torch.randn(4, 3, 224, 224, device=device)
    hr = torch.randn(4, 3, 224, 224, device=device)
    loss = loss_fn(sr, hr)
    assert loss.dim() == 0, "Loss should still return a scalar for batched input"


# ---- Preprocessing fixture and test ---- #
@pytest.fixture
def vgg_preprocessed_images(device):
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img1 = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    img2 = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))

    tensor1 = transform(img1).unsqueeze(0).to(device)
    tensor2 = transform(img2).unsqueeze(0).to(device)
    return tensor1, tensor2


def test_with_vgg_preprocessed_images(loss_fn, vgg_preprocessed_images):
    sr, hr = vgg_preprocessed_images
    loss = loss_fn(sr, hr)
    assert loss.dim() == 0
    assert loss.item() >= 0.0, "Loss should be non-negative"
