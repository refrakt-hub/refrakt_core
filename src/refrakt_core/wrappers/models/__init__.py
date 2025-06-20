from .convnext import ConvNeXtWrapper
from .msn import MSNWrapper
from .resnet import ResNetWrapper
from .srgan import SRGANWrapper
from .autoencoder import AutoencoderWrapper
from .mae import MAEWrapper
from .dino import DINOWrapper
from .simclr import SimCLRWrapper
from .vit import ViTWrapper
from .swin import SwinTransformerWrapper

__all__ = [
    "ConvNeXtWrapper",
    "MSNWrapper",
    "ResNetWrapper",
    "SRGANWrapper",
    "AutoencoderWrapper",
    "MAEWrapper",
    "DINOWrapper",
    "SimCLRWrapper",
    "ViTWrapper",
    "SwinTransformerWrapper"
]