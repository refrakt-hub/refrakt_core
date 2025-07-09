"""
Swin Transformer model for image classification.

Implements a hierarchical vision transformer using shifted windows, with
custom embedding, patch merging, and stacked Swin stages.
"""

from torch import Tensor, nn

from refrakt_core.registry.model_registry import register_model
from refrakt_core.utils.classes.embedding import Embedding
from refrakt_core.utils.classes.swin import AlternateSwin
from refrakt_core.utils.classes.utils import Merge


@register_model("swin")
class SwinTransformer(nn.Module):
    """
    Swin Transformer for hierarchical vision processing.

    This model implements the Swin architecture with multiple stages and
    hierarchical feature merging. It supports image classification tasks and
    returns logits from a final linear head.
    """

    embedding: Embedding
    patch1: Merge
    patch2: Merge
    patch3: Merge
    stage1: AlternateSwin
    stage2: AlternateSwin
    stage3_1: AlternateSwin
    stage3_2: AlternateSwin
    stage3_3: AlternateSwin
    stage4: AlternateSwin
    avgpool: nn.AdaptiveAvgPool2d
    head: nn.Linear

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        """
        Initialize the Swin Transformer model.

        Args:
            in_channels: Number of input channels (e.g. 3 for RGB).
            num_classes: Number of output classes for classification.
        """
        super().__init__()
        self.embedding = Embedding(in_channels=in_channels)
        self.patch1 = Merge(96)
        self.patch2 = Merge(192)
        self.patch3 = Merge(384)

        self.stage1 = AlternateSwin(96, 3)
        self.stage2 = AlternateSwin(192, 6)
        self.stage3_1 = AlternateSwin(384, 12)
        self.stage3_2 = AlternateSwin(384, 12)
        self.stage3_3 = AlternateSwin(384, 12)
        self.stage4 = AlternateSwin(768, 24)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(768, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Swin Transformer.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output logits of shape (B, num_classes).
        """
        x = self.embedding(x)
        x = self.patch1(self.stage1(x))
        x = self.patch2(self.stage2(x))
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        x = self.stage3_3(x)
        x = self.patch3(x)
        x = self.stage4(x)

        x = x.mean(dim=1)  # global average pooling over patch tokens
        x = self.head(x)
        return x  # type: ignore[no-any-return]
