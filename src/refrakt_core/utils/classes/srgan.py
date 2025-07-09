"""
The class implementations for the modules inside SRGAN.

Modules:
- UpsampleBlock
- SRResidualBlock
- Generator
- Discriminator
"""

import math
from typing import List, cast

import torch
from torch import Tensor, nn

from refrakt_core.utils.classes.resnet import ResidualBlock


class UpsampleBlock(nn.Module):
    """
    Optimized upsampling block using ConvTranspose2d.
    """

    def __init__(
        self, in_channels: int, out_channels: int, scale_factor: int = 2
    ) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels * scale_factor**2, kernel_size=3, padding=1
            ),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the Upsample block"""
        return cast(Tensor, self.upsample(x))


class SRResidualBlock(ResidualBlock):
    """
    A modified version of ResidualBlock specifically for Super Resolution.
    """

    def __init__(self, channels: int) -> None:
        nn.Module.__init__(self)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        return cast(Tensor, x + res)


class Generator(nn.Module):
    """
    SRGAN Generator.
    """

    def __init__(self, scale_factor: int = 4) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        upsample_num = int(math.log(scale_factor, 2))

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU()
        )
        self.res_blocks = nn.Sequential(*(SRResidualBlock(64) for _ in range(5)))
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )
        upsample_blocks: List[nn.Module] = [
            UpsampleBlock(64, 64) for _ in range(upsample_num)
        ]
        upsample_blocks.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.upsample_blocks = nn.Sequential(*upsample_blocks)

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward method for computing the generator output.
        """
        block1 = self.block1(x)
        res_blocks = self.res_blocks(block1)
        final = self.final(res_blocks)
        output = block1 + final
        output = self.upsample_blocks(output)
        return (torch.tanh(output) + 1) / 2


class Discriminator(nn.Module):
    """
    SRGAN Discriminator.
    """

    def __init__(self) -> None:
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward method that gives a sigmoid output for the discriminator.
        """
        batch = x.size(0)
        return torch.sigmoid(self.disc(x).view(batch))
