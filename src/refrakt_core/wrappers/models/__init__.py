from .convnext import ConvNeXtWrapper
from .msn import MSNWrapper
from .resnet import ResNetWrapper
from .srgan import SRGANWrapper
from .autoencoder import AutoencoderWrapper
from .mae import MAEWrapper

__all__ = [
    "ConvNeXtWrapper",
    "MSNWrapper",
    "ResNetWrapper",
    "SRGANWrapper",
    "AutoencoderWrapper",
    "MAEWrapper",
]