import pytest
import tempfile
import torch

from refrakt_core.models.srgan import SRGAN

# Add device fixture
@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_generator_output_shape(device):
    model = SRGAN(scale_factor=4).to(device)
    model.generator.eval()
    input_tensor = torch.randn(1, 3, 48, 48).to(device)
    with torch.no_grad():
        output = model.generator(input_tensor)
    expected_shape = (1, 3, 192, 192)
    assert output.shape == expected_shape

def test_discriminator_output_shape(device):
    model = SRGAN(scale_factor=4).to(device)
    model.discriminator.eval()
    input_tensor = torch.randn(1, 3, 192, 192).to(device)
    with torch.no_grad():
        output = model.discriminator(input_tensor)
    assert output.shape == (1,)

def test_generate_function(device):
    model = SRGAN(scale_factor=4).to(device)
    input_tensor = torch.randn(1, 3, 48, 48).to(device)
    output = model.generate(input_tensor)
    assert output.shape == (1, 3, 192, 192)
    assert (0 <= output).all() and (output <= 1).all()

def test_discriminate_function(device):
    model = SRGAN(scale_factor=4).to(device)
    input_tensor = torch.randn(1, 3, 192, 192).to(device)
    output = model.discriminate(input_tensor)
    assert output.shape == (1,)
    assert (0 <= output).all() and (output <= 1).all()

def test_model_save_and_load(device):
    model = SRGAN(scale_factor=4).to(device)
    input_tensor = torch.randn(1, 3, 48, 48).to(device)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        path = tmp.name
        model.save_model(path)
        new_model = SRGAN(scale_factor=4)
        new_model.load_model(path)
        # Move loaded model to device
        new_model = new_model.to(device)
        output = new_model.generate(input_tensor)
        assert output.shape == (1, 3, 192, 192)