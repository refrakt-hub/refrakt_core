"""
ResNet model definitions for Refrakt framework.

Includes ResNet18, ResNet50, ResNet101, and ResNet152 with support for custom
block definitions (Basic and Bottleneck), custom input channels, and projection layers.
"""

from typing import Any, List, Optional, Type

import torch
from torch import Tensor, nn

from refrakt_core.models.templates.models import BaseClassifier
from refrakt_core.registry.model_registry import register_model
from refrakt_core.utils.classes.resnet import BottleneckBlock, ResidualBlock


class ResNet(BaseClassifier):
    """
    ResNet base class for deep residual networks.

    Args:
        block (nn.Module): Residual block type (e.g., BasicBlock or BottleneckBlock).
        layers (list): Number of blocks in each layer.
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        model_name (str): Model name for registration and tracking.
    """

    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        in_channels: int = 3,
        num_classes: int = 10,
        model_name: str = "resnet",
    ) -> None:
        super().__init__(num_classes=num_classes, model_name=model_name)

        self.inplanes: int = 64
        self.conv1: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool: nn.Module = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0: nn.Sequential = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1: nn.Sequential = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2: nn.Sequential = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3: nn.Sequential = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim: int = 512  # Required by DINOModel
        self.projection: nn.Module = nn.Identity()
        self.fc: nn.Module = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(
        self, block: Type[nn.Module], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample: Optional[nn.Sequential] = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )

        layers: List[nn.Module] = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(
        self, x: Tensor, return_features: bool = False, **kwargs: Any
    ) -> torch.Tensor:
        # print("[DEBUG ResNet] return_features:", return_features)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if return_features:
            return x  # type: ignore[no-any-return]
        return self.fc(x)  # type: ignore[no-any-return]


@register_model("resnet18")
class ResNet18(ResNet):
    """ResNet-18 variant using basic residual blocks."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__(
            block=ResidualBlock,
            layers=[2, 2, 2, 2],
            in_channels=in_channels,
            num_classes=num_classes,
        )


@register_model("resnet50")
class ResNet50(ResNet):
    """ResNet-50 variant using bottleneck blocks."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__(
            block=BottleneckBlock,
            layers=[3, 4, 6, 3],
            in_channels=in_channels,
            num_classes=num_classes,
        )


@register_model("resnet101")
class ResNet101(ResNet):
    """ResNet-101 variant using bottleneck blocks."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__(
            block=BottleneckBlock,
            layers=[3, 4, 23, 3],
            in_channels=in_channels,
            num_classes=num_classes,
        )


@register_model("resnet152")
class ResNet152(ResNet):
    """ResNet-152 variant using bottleneck blocks."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__(
            block=BottleneckBlock,
            layers=[3, 8, 36, 3],
            in_channels=in_channels,
            num_classes=num_classes,
        )
