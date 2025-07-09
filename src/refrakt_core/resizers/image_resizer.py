"""
Improved image resizing system for Refrakt with size validation and standard sizes.

This module provides a flexible system for resizing images with validation
and support for standard image sizes to prevent information loss.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


@dataclass
class ImageSizeConfig:
    """Configuration for image size constraints."""

    standard_size: Tuple[int, int] = (224, 224)
    max_size: Tuple[int, int] = (448, 448)  # 2x standard size

    min_size: Tuple[int, int] = (28, 28)
    aspect_ratio_tolerance: float = 0.1
    interpolation_method: str = "lanczos"  # lanczos, bilinear, nearest


class ImageSizeValidator:
    """
    Validates and manages image sizes with configurable constraints.

    This class provides validation for image sizes and can throw exceptions
    for images that are too large to resize without significant information loss.
    """

    def __init__(self, config: Optional[ImageSizeConfig] = None):
        """
        Initialize the image size validator.

        Args:
            config: Configuration for size constraints
        """
        self.config = config or ImageSizeConfig()
        self._setup_interpolation()

    def _setup_interpolation(self) -> None:
        """Setup interpolation methods for different backends."""
        self.pil_interpolation = {
            "lanczos": Image.Resampling.LANCZOS,
            "bilinear": Image.Resampling.BILINEAR,
            "nearest": Image.Resampling.NEAREST,
            "bicubic": Image.Resampling.BICUBIC,
        }

        self.torch_interpolation = {
            "lanczos": "lanczos",
            "bilinear": "bilinear",
            "nearest": "nearest",
            "bicubic": "bicubic",
        }

    def validate_image_size(
        self, size: Tuple[int, int], image_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if an image size is acceptable.

        Args:
            size: Image size (width, height)
            image_path: Optional path for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        width, height = size

        # Check minimum size
        if width < self.config.min_size[0] or height < self.config.min_size[1]:
            error_msg = (
                f"Image {'at ' + str(image_path) if image_path else ''} is too small "
                f"({width}x{height}). Minimum size is {self.config.min_size[0]}x{self.config.min_size[1]}."
            )
            return False, error_msg

        # Check maximum size
        if width > self.config.max_size[0] or height > self.config.max_size[1]:
            error_msg = (
                f"Image {'at ' + str(image_path) if image_path else ''} is too large "
                f"({width}x{height}). Maximum size is {self.config.max_size[0]}x{self.config.max_size[1]}. "
                f"Please resize the image to {self.config.standard_size[0]}x{self.config.standard_size[1]} or smaller."
            )
            return False, error_msg

        return True, None

    def validate_aspect_ratio(
        self, size: Tuple[int, int], target_ratio: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if an image's aspect ratio is acceptable.

        Args:
            size: Image size (width, height)
            target_ratio: Target aspect ratio (width/height)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if target_ratio is None:
            target_ratio = self.config.standard_size[0] / self.config.standard_size[1]

        current_ratio = size[0] / size[1]
        ratio_diff = abs(current_ratio - target_ratio)

        if ratio_diff > self.config.aspect_ratio_tolerance:
            error_msg = (
                f"Image aspect ratio {current_ratio:.2f} differs too much from target "
                f"{target_ratio:.2f}. Maximum difference allowed: {self.config.aspect_ratio_tolerance}"
            )
            return False, error_msg

        return True, None


class SmartImageResizer:
    """
    Smart image resizer that maintains aspect ratio and handles different resize strategies.

    This class provides intelligent resizing that can handle different scenarios
    while maintaining image quality and aspect ratios.
    """

    def __init__(self, config: Optional[ImageSizeConfig] = None):
        """
        Initialize the smart image resizer.

        Args:
            config: Configuration for resizing behavior
        """
        self.config = config or ImageSizeConfig()
        self.validator = ImageSizeValidator(config)
        self._setup_interpolation()

    def _setup_interpolation(self) -> None:
        """Setup interpolation methods."""
        self.pil_interpolation = {
            "lanczos": Image.Resampling.LANCZOS,
            "bilinear": Image.Resampling.BILINEAR,
            "nearest": Image.Resampling.NEAREST,
            "bicubic": Image.Resampling.BICUBIC,
        }

    def resize_pil_image(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        strategy: str = "maintain_aspect",
    ) -> Image.Image:
        """
        Resize a PIL image using the specified strategy.

        Args:
            image: PIL Image to resize
            target_size: Target size (width, height). If None, uses standard size
            strategy: Resize strategy ('maintain_aspect', 'crop', 'pad', 'stretch')

        Returns:
            Resized PIL Image
        """
        if target_size is None:
            target_size = self.config.standard_size

        # Validate original size
        original_size = image.size
        is_valid, error_msg = self.validator.validate_image_size(original_size)
        if not is_valid:
            raise ValueError(error_msg)

        if strategy == "maintain_aspect":
            return self._resize_maintain_aspect(image, target_size)
        elif strategy == "crop":
            return self._resize_crop(image, target_size)
        elif strategy == "pad":
            return self._resize_pad(image, target_size)
        elif strategy == "stretch":
            return self._resize_stretch(image, target_size)
        else:
            raise ValueError(f"Unknown resize strategy: {strategy}")

    def _resize_maintain_aspect(
        self, image: Image.Image, target_size: Tuple[int, int]
    ) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        target_width, target_height = target_size
        original_width, original_height = image.size

        # Calculate scaling factors
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)

        # Calculate new size
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image
        resized = image.resize(
            (new_width, new_height),
            self.pil_interpolation[self.config.interpolation_method],
        )

        # Create new image with target size and paste resized image
        result = Image.new(
            image.mode, target_size, (0, 0, 0) if image.mode == "RGB" else 0
        )

        # Center the resized image
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        result.paste(resized, (paste_x, paste_y))

        return result

    def _resize_crop(
        self, image: Image.Image, target_size: Tuple[int, int]
    ) -> Image.Image:
        """Resize image by cropping to target size."""
        target_width, target_height = target_size
        original_width, original_height = image.size

        # Calculate scaling factors
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = max(scale_w, scale_h)

        # Calculate new size
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image
        resized = image.resize(
            (new_width, new_height),
            self.pil_interpolation[self.config.interpolation_method],
        )

        # Crop to target size
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        return resized.crop((left, top, right, bottom))

    def _resize_pad(
        self, image: Image.Image, target_size: Tuple[int, int]
    ) -> Image.Image:
        """Resize image by padding to target size."""
        target_width, target_height = target_size
        original_width, original_height = image.size

        # Calculate scaling factors
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)

        # Calculate new size
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image
        resized = image.resize(
            (new_width, new_height),
            self.pil_interpolation[self.config.interpolation_method],
        )

        # Create new image with target size and paste resized image
        result = Image.new(
            image.mode, target_size, (0, 0, 0) if image.mode == "RGB" else 0
        )

        # Center the resized image
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        result.paste(resized, (paste_x, paste_y))

        return result

    def _resize_stretch(
        self, image: Image.Image, target_size: Tuple[int, int]
    ) -> Image.Image:
        """Resize image by stretching to target size."""
        return image.resize(
            target_size, self.pil_interpolation[self.config.interpolation_method]
        )

    def resize_tensor(
        self,
        tensor: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        strategy: str = "maintain_aspect",
    ) -> torch.Tensor:
        """
        Resize a tensor image using the specified strategy.

        Args:
            tensor: Tensor image (C, H, W) or (B, C, H, W)
            target_size: Target size (height, width). If None, uses standard size
            strategy: Resize strategy

        Returns:
            Resized tensor
        """
        if target_size is None:
            target_size = (
                self.config.standard_size[1],
                self.config.standard_size[0],
            )  # (H, W)

        # Handle batch dimension
        if tensor.dim() == 4:
            batch_size = tensor.size(0)
            resized_tensors = []
            for i in range(batch_size):
                resized = self._resize_single_tensor(tensor[i], target_size, strategy)
                resized_tensors.append(resized)
            return torch.stack(resized_tensors)
        else:
            return self._resize_single_tensor(tensor, target_size, strategy)

    def _resize_single_tensor(
        self, tensor: torch.Tensor, target_size: Tuple[int, int], strategy: str
    ) -> torch.Tensor:
        """Resize a single tensor image."""
        if strategy == "stretch":
            out = F.interpolate(
                tensor.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            assert isinstance(out, torch.Tensor)
            return out
        else:
            # For other strategies, convert to PIL, resize, then convert back
            # This is a simplified approach - in practice, you might want more sophisticated tensor handling
            pil_image = T.ToPILImage()(tensor)
            resized_pil = self.resize_pil_image(
                pil_image, (target_size[1], target_size[0]), strategy
            )
            out = T.ToTensor()(resized_pil)
            assert isinstance(out, torch.Tensor)
            return out


class StandardImageTransform:
    """
    Standard image transform that includes resizing and validation.

    This class provides a complete transform pipeline that includes
    size validation, resizing, and other standard preprocessing steps.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        resize_strategy: str = "maintain_aspect",
        normalize: bool = True,
        augment: bool = False,
    ):
        """
        Initialize the standard image transform.

        Args:
            target_size: Target image size (width, height)
            resize_strategy: Strategy for resizing
            normalize: Whether to normalize the image
            augment: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.resize_strategy = resize_strategy
        self.normalize = normalize
        self.augment = augment

        self.resizer = SmartImageResizer(ImageSizeConfig(standard_size=target_size))
        self.validator = ImageSizeValidator(ImageSizeConfig(standard_size=target_size))

        self._setup_transforms()

    def _setup_transforms(self) -> None:
        """Setup the transform pipeline."""
        transforms_list = []

        # Resize transform
        transforms_list.append(
            lambda img: self.resizer.resize_pil_image(
                img, self.target_size, self.resize_strategy
            )
        )

        # Convert to tensor
        transforms_list.append(T.ToTensor())

        # Normalize if requested
        if self.normalize:
            transforms_list.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        # Data augmentation if requested
        if self.augment:
            augmentation_transforms = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
            # Insert augmentation before normalization
            transforms_list = (
                transforms_list[:-1] + augmentation_transforms + transforms_list[-1:]
            )

        self.transform = T.Compose(transforms_list)

    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Apply the transform to an image.

        Args:
            image: PIL Image or tensor to transform

        Returns:
            Transformed tensor
        """
        if isinstance(image, torch.Tensor):
            # Validate tensor size
            if image.dim() == 3:  # (C, H, W)
                size = (image.size(2), image.size(1))  # (W, H)
            else:  # (H, W)
                size = (image.size(1), image.size(0))  # (W, H)

            is_valid, error_msg = self.validator.validate_image_size(size)
            if not is_valid:
                raise ValueError(error_msg)

            # Convert tensor to PIL for consistent processing
            if image.dim() == 3:
                image = T.ToPILImage()(image)
            else:
                image = T.ToPILImage()(image.unsqueeze(0))

        # Apply transform
        result = self.transform(image)
        assert isinstance(result, torch.Tensor)
        return result

    def validate_image(
        self, image_path: Union[str, Path]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate an image file for size constraints.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with Image.open(image_path) as img:
                return self.validator.validate_image_size(img.size, image_path)
        except Exception as e:
            return False, f"Failed to validate image {image_path}: {e}"

    def resize_image(self, image: Any) -> torch.Tensor:
        result = self.transform(image)
        if isinstance(result, torch.Tensor):
            return result
        elif isinstance(result, Image.Image):
            tensor_result = T.ToTensor()(result)
            assert isinstance(tensor_result, torch.Tensor)
            return tensor_result
        else:
            raise TypeError(
                "resize_image must return a torch.Tensor or Image.Image convertible to tensor."
            )

    def to_tensor(self, resized_pil: Image.Image) -> torch.Tensor:
        out = T.ToTensor()(resized_pil)
        assert isinstance(out, torch.Tensor)
        return out
