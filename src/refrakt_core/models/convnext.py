"""
ConvNeXt-based image classification model with hierarchical ConvNeXt blocks.

Implements a simplified ConvNeXt architecture with layer norm and GELU.
"""

from torch import Tensor, nn

from refrakt_core.models.templates.models import BaseClassifier
from refrakt_core.registry.model_registry import register_model


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style convolutional block with two Conv2D layers,
    GELU activation, and LayerNorm in between.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        stride (int): Stride for convolution.
        padding (int): Padding for convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.conv2: nn.Conv2d = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding
        )
        self.ln: nn.LayerNorm = nn.LayerNorm(out_channels)
        self.gelu: nn.GELU = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the ConvNeXt block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor after convolutions and activation
        """
        out = self.conv1(x)
        out = self.gelu(out)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = self.ln(out)
        out = out.permute(0, 3, 1, 2)  # back to (B, C, H, W)
        out = self.conv2(out)
        return out  # type: ignore[no-any-return]


@register_model("convnext")
class ConvNeXt(BaseClassifier):
    """
    ConvNeXt-based image classification model.

    Args:
        in_channels (int): Number of input image channels.
        num_classes (int): Number of target classes.
        model_name (str): Name used for registration/logging.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        model_name: str = "convnext",
    ) -> None:
        super().__init__(num_classes=num_classes, model_name=model_name)
        self.stem: nn.Conv2d = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
        self.block1: ConvNeXtBlock = ConvNeXtBlock(96, 192)
        self.block2: ConvNeXtBlock = ConvNeXtBlock(192, 384)
        self.block3: ConvNeXtBlock = ConvNeXtBlock(384, 768)
        self.gap: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.fc: nn.Linear = nn.Linear(768, num_classes)

    def forward(self, x: Tensor, return_features: bool = False) -> Tensor:
        """
        Forward pass through ConvNeXt model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            return_features (bool): If True, return features before fc layer.

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes) or features if return_features is True
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        if return_features:
            return x  # features before fc
        x = self.fc(x)
        return x  # logits
