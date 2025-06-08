import pytest
import torch
import torch.nn as nn

from refrakt_core.models.resnet import ResidualBlock, ResNet

def get_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_resnet(device):
    """Create a small ResNet for testing."""
    model = ResNet(
        block=ResidualBlock, layers=[1, 1, 1, 1], in_channels=3, num_classes=10
    )
    model = model.to(device)
    # Ensure model is in eval mode for consistent behavior
    model.eval()
    return model


def test_init(small_resnet):
    """Test that the ResNet initializes correctly."""
    assert small_resnet.model_name == "resnet"
    assert small_resnet.model_type == "classifier"
    assert small_resnet.num_classes == 10


def test_residual_block():
    """Test the ResidualBlock."""
    block = ResidualBlock(in_channels=64, out_channels=64)
    x = torch.randn(1, 64, 56, 56)
    output = block(x)
    assert output.shape == (1, 64, 56, 56)

    # Test with downsample
    block_with_downsample = ResidualBlock(
        in_channels=64,
        out_channels=128,
        stride=2,
        downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2), nn.BatchNorm2d(128)
        ),
    )
    output = block_with_downsample(x)
    assert output.shape == (1, 128, 28, 28)


def test_forward(small_resnet):
    device = get_device(small_resnet)
    x = torch.randn(2, 3, 224, 224).to(device)
    output = small_resnet(x)
    assert output.shape == (2, 10)


def test_predict(small_resnet):
    """Test the predict method."""
    device = get_device(small_resnet)
    
    # Ensure model is properly on device
    small_resnet = small_resnet.to(device)
    
    # Create input tensor on same device as model
    x = torch.randn(2, 3, 224, 224, device=device)
    
    # Use torch.no_grad() to avoid potential gradient computation issues
    with torch.no_grad():
        predictions = small_resnet.predict(x)
    
    assert predictions.shape == (2,)
    assert predictions.dtype == torch.int64

    with torch.no_grad():
        probabilities = small_resnet.predict(x, return_probs=True)
    
    assert probabilities.shape == (2, 10)
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2, device=device))


def test_predict_proba(small_resnet):
    """Test the predict_proba method."""
    device = get_device(small_resnet)
    
    # Ensure model is properly on device
    small_resnet = small_resnet.to(device)
    
    # Create input tensor on same device as model
    x = torch.randn(2, 3, 224, 224, device=device)
    
    # Use torch.no_grad() to avoid potential gradient computation issues
    with torch.no_grad():
        probabilities = small_resnet.predict_proba(x)
    
    assert probabilities.shape == (2, 10)
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2, device=device))


def test_model_on_different_input_sizes(small_resnet):
    """Test that the model works with different input sizes."""
    device = get_device(small_resnet)
    
    # Create tensors directly on the target device
    x_small = torch.randn(2, 3, 160, 160, device=device)
    output_small = small_resnet(x_small)
    assert output_small.shape == (2, 10)

    x_large = torch.randn(2, 3, 256, 256, device=device)
    output_large = small_resnet(x_large)
    assert output_large.shape == (2, 10)


def test_save_load(small_resnet, tmp_path):
    """Test save and load functionality."""
    device = get_device(small_resnet)
    save_path = tmp_path / "resnet.pt"

    # Create input tensor directly on device
    x = torch.randn(2, 3, 224, 224, device=device)
    
    with torch.no_grad():
        original_predictions = small_resnet.predict(x)

    # Save the model
    small_resnet.save_model(str(save_path))

    # Create a new model and load the saved weights
    new_model = ResNet(
        block=ResidualBlock, layers=[1, 1, 1, 1], in_channels=3, num_classes=10
    )
    new_model.load_model(str(save_path))
    new_model = new_model.to(device)
    new_model.eval()
    
    with torch.no_grad():
        loaded_predictions = new_model.predict(x)
    
    assert torch.all(original_predictions == loaded_predictions)

    assert new_model.model_name == small_resnet.model_name
    assert new_model.model_type == small_resnet.model_type
    assert new_model.num_classes == small_resnet.num_classes