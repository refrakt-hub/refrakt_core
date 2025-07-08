"""
Residual and Vision Transformer (ViT) blocks used in the Refrakt framework.

Includes:
- Standard ResidualBlock (like ResNet)
- BottleneckBlock with expansion
- SkipConnections wrapper with LayerNorm and dropout
- ViTResidual block (Norm -> MHSA -> Norm -> MLP)
"""

from typing import Callable, Optional, cast

from torch import Tensor, nn

from refrakt_core.utils.classes.attention import MSA
from refrakt_core.utils.classes.utils import LayerNormalization


class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet-style architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride value for convolution. Default is 1.
        downsample (Optional[nn.Module]): Downsampling layer if needed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the residual block.
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return cast(Tensor, out)


class BottleneckBlock(nn.Module):
    """
    Bottleneck block with 3 convolutions and expansion factor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of intermediate output channels.
        stride (int): Stride for the second convolution.
        downsample (Optional[nn.Module]): Optional downsampling module.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels * self.expansion, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the bottleneck block.
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return cast(Tensor, out)


class SkipConnections(nn.Module):
    """
    Adds a skip connection around a sublayer with LayerNormalization and dropout.

    Args:
        dropout (float): Dropout probability.
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        """
        Applies normalized sublayer with residual skip.

        Args:
            x (Tensor): Input tensor.
            sublayer (Callable): A function or module applied to normalized input.

        Returns:
            Tensor: Output with residual connection.
        """
        y = self.norm(x)
        y = sublayer(y)
        y = self.dropout(y)
        return cast(Tensor, x + y)


class ViTResidual(nn.Module):
    """
    Residual block for Vision Transformers (ViT) using MHSA and MLP.

    Args:
        hidden_d (int): Dimension of hidden features.
        n_heads (int): Number of attention heads.
        mlp_ratio (int): Multiplier for MLP hidden size.
    """

    def __init__(self, hidden_d: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies MHSA and MLP with residual connections.

        Args:
            x (Tensor): Input tensor of shape (B, N, D)

        Returns:
            Tensor: Output tensor.
        """
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return cast(Tensor, out)
