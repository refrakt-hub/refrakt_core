from .ce import CrossEntropyLossWrapper
from .dino import DINOLossWrapper
from .gan import GANLossWrapper
from .mae import MAELossWrapper
from .mse import MSELossWrapper
from .msn import MSNLossWrapper
from .ntxent import NTXentLossWrapper
from .perceptual import PerceptualLossWrapper
from .vae import VAELossWrapper

__all__ = [
    "CrossEntropyLossWrapper",
    "DINOLossWrapper",
    "GANLossWrapper",
    "MAELossWrapper",
    "MSELossWrapper",
    "MSNLossWrapper",
    "NTXentLossWrapper",
    "PerceptualLossWrapper",
    "VAELossWrapper",
    "NTXentLossWrapper",
]
