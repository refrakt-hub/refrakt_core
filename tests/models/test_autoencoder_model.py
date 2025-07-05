import pytest
import torch

from refrakt_core.models.autoencoder import AutoEncoder


# Define device fixture at module level
@pytest.fixture(scope="module")
def test_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_autoencoder(test_device):
    return AutoEncoder(input_dim=784, hidden_dim=8, mode="simple").to(test_device)


@pytest.fixture
def vae_autoencoder(test_device):
    return AutoEncoder(input_dim=784, hidden_dim=8, mode="vae").to(test_device)


def test_simple_autoencoder_init(simple_autoencoder):
    """Test that the simple autoencoder initializes correctly."""
    assert simple_autoencoder.input_dim == 784
    assert simple_autoencoder.hidden_dim == 8
    assert simple_autoencoder.mode == "simple"
    assert simple_autoencoder.model_name == "autoencoder"
    assert simple_autoencoder.model_type == "autoencoder"


def test_vae_init(vae_autoencoder):
    """Test that the variational autoencoder initializes correctly."""
    assert vae_autoencoder.input_dim == 784
    assert vae_autoencoder.hidden_dim == 8
    assert vae_autoencoder.mode == "vae"
    assert hasattr(vae_autoencoder, "mu")
    assert hasattr(vae_autoencoder, "sigma")


def test_simple_encoder(simple_autoencoder):
    device = next(simple_autoencoder.parameters()).device
    x = torch.randn(10, 784, device=device)
    encoded = simple_autoencoder.encode(x)
    assert encoded.shape == (10, 8)


def test_vae_encoder(vae_autoencoder):
    device = next(vae_autoencoder.parameters()).device
    x = torch.randn(10, 784, device=device)
    mu, sigma = vae_autoencoder.encode(x)
    assert mu.shape == (10, 8)
    assert sigma.shape == (10, 8)


def test_reparameterize(vae_autoencoder):
    device = next(vae_autoencoder.parameters()).device
    mu = torch.zeros(10, 8, device=device)
    sigma = torch.zeros(10, 8, device=device)
    z = vae_autoencoder._reparameterize(mu, sigma)
    assert z.shape == (10, 8)
    assert torch.abs(z.mean()) < 0.5
    assert torch.abs(z.std() - 1) < 0.5


def test_decoder(simple_autoencoder):
    device = next(simple_autoencoder.parameters()).device
    z = torch.randn(10, 8, device=device)
    decoded = simple_autoencoder.decode(z)
    assert decoded.shape == (10, 784)
    assert torch.all(decoded >= 0) and torch.all(decoded <= 1)


def test_simple_forward(simple_autoencoder):
    device = next(simple_autoencoder.parameters()).device
    x = torch.randn(10, 784, device=device)
    output = simple_autoencoder(x)
    assert output.shape == (10, 784)


def test_vae_forward(vae_autoencoder):
    device = next(vae_autoencoder.parameters()).device
    x = torch.randn(10, 784, device=device)
    output = vae_autoencoder(x)
    recon = output["recon"]
    mu = output["mu"]
    logvar = output["logvar"]
    assert recon.shape == (10, 784)
    assert mu.shape == (10, 8)
    assert logvar.shape == (10, 8)


def test_get_latent(simple_autoencoder):
    device = next(simple_autoencoder.parameters()).device
    x = torch.randn(10, 784, device=device)
    latent = simple_autoencoder.get_latent(x)
    assert latent.shape == (10, 8)


def test_invalid_type():
    """Test that an invalid type raises ValueError."""
    with pytest.raises(ValueError):
        model = AutoEncoder(input_dim=784, hidden_dim=8, mode="invalid")
        x = torch.randn(10, 784)
        model(x)
