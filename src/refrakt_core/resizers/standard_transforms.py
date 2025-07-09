"""
Standard transforms for Refrakt with image resizing and size validation.

This module provides standard transform pipelines with image resizing capabilities
and size validation to prevent information loss.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
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


def validate_image_size(
    size: Tuple[int, int],
    image_path: Optional[Union[str, Path]] = None,
    max_size: Tuple[int, int] = (448, 448),
    min_size: Tuple[int, int] = (28, 28),
) -> Tuple[bool, Optional[str]]:
    """
    Validate if an image size is acceptable.

    Args:
        size: Image size (width, height)
        image_path: Optional path for error messages
        max_size: Maximum allowed image size
        min_size: Minimum allowed image size

    Returns:
        Tuple of (is_valid, error_message)
    """
    width, height = size

    # Check minimum size
    if width < min_size[0] or height < min_size[1]:
        error_msg = (
            f"Image {'at ' + str(image_path) if image_path else ''} is too small "
            f"({width}x{height}). Minimum size is {min_size[0]}x{min_size[1]}."
        )
        return False, error_msg

    # Check maximum size
    if width > max_size[0] or height > max_size[1]:
        error_msg = (
            f"Image {'at ' + str(image_path) if image_path else ''} is too large "
            f"({width}x{height}). Maximum size is {max_size[0]}x{max_size[1]}. "
            f"Please resize the image to {max_size[0]//2}x{max_size[1]//2} or smaller."
        )
        return False, error_msg

    return True, None


def resize_image_maintain_aspect(
    image: Image.Image, target_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Resize image maintaining aspect ratio.

    Args:
        image: PIL Image to resize
        target_size: Target size (width, height)

    Returns:
        Resized PIL Image
    """
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
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create new image with target size and paste resized image
    result = Image.new(image.mode, target_size, (0, 0, 0) if image.mode == "RGB" else 0)

    # Center the resized image
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    result.paste(resized, (paste_x, paste_y))

    return result


def resize_image_crop(
    image: Image.Image, target_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Resize image by cropping to target size.

    Args:
        image: PIL Image to resize
        target_size: Target size (width, height)

    Returns:
        Resized PIL Image
    """
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
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Crop to target size
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return resized.crop((left, top, right, bottom))


def resize_image_stretch(
    image: Image.Image, target_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Resize image by stretching to target size.

    Args:
        image: PIL Image to resize
        target_size: Target size (width, height)

    Returns:
        Resized PIL Image
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)


def create_standard_transform(
    target_size: Tuple[int, int] = (224, 224),
    resize_strategy: str = "maintain_aspect",
    normalize: bool = True,
    augment: bool = False,
    max_size: Tuple[int, int] = (448, 448),
) -> T.Compose:
    """
    Create a standard transform pipeline.

    Args:
        target_size: Target image size (width, height)
        resize_strategy: Strategy for resizing ('maintain_aspect', 'crop', 'stretch')
        normalize: Whether to normalize the image
        augment: Whether to apply data augmentation
        max_size: Maximum allowed image size

    Returns:
        Transform pipeline
    """
    transforms_list = []

    # Resize transform
    if resize_strategy == "maintain_aspect":

        def resize_fn(img: Image.Image) -> Image.Image:
            return resize_image_maintain_aspect(img, target_size)

    elif resize_strategy == "crop":

        def resize_fn(img: Image.Image) -> Image.Image:
            return resize_image_crop(img, target_size)

    elif resize_strategy == "stretch":

        def resize_fn(img: Image.Image) -> Image.Image:
            return resize_image_stretch(img, target_size)

    else:
        raise ValueError(f"Unknown resize strategy: {resize_strategy}")

    transforms_list.append(resize_fn)

    # Convert to tensor
    transforms_list.append(T.ToTensor())

    # Normalize if requested
    if normalize:
        transforms_list.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    # Data augmentation if requested
    if augment:
        augmentation_transforms = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
        # Insert augmentation before normalization
        transforms_list = (
            transforms_list[:-1] + augmentation_transforms + transforms_list[-1:]
        )

    return T.Compose(transforms_list)


def create_classification_transform(
    target_size: Tuple[int, int] = (224, 224),
    augment: bool = True,
    normalize: bool = True,
) -> T.Compose:
    """
    Create a transform for classification tasks.

    Args:
        target_size: Target image size
        augment: Whether to apply augmentation
        normalize: Whether to normalize

    Returns:
        Transform pipeline
    """
    return create_standard_transform(
        target_size=target_size,
        resize_strategy="maintain_aspect",
        normalize=normalize,
        augment=augment,
    )


def create_contrastive_transform(
    target_size: Tuple[int, int] = (224, 224), normalize: bool = True
) -> T.Compose:
    """
    Create a transform for contrastive learning tasks.

    Args:
        target_size: Target image size
        normalize: Whether to normalize

    Returns:
        Transform pipeline
    """
    return create_standard_transform(
        target_size=target_size,
        resize_strategy="maintain_aspect",
        normalize=normalize,
        augment=False,  # Contrastive learning handles augmentation separately
    )


def create_gan_transform(
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = False,  # GANs often don't use normalization
) -> T.Compose:
    """
    Create a transform for GAN tasks.

    Args:
        target_size: Target image size
        normalize: Whether to normalize

    Returns:
        Transform pipeline
    """
    return create_standard_transform(
        target_size=target_size,
        resize_strategy="maintain_aspect",
        normalize=normalize,
        augment=False,
    )


def validate_transform_input(
    image: Union[Image.Image, torch.Tensor], max_size: Tuple[int, int] = (448, 448)
) -> None:
    """
    Validate input for transforms.

    Args:
        image: Image to validate
        max_size: Maximum allowed size

    Raises:
        ValueError: If image is too large
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:  # (C, H, W)
            size = (image.size(2), image.size(1))  # (W, H)
        else:  # (H, W)
            size = (image.size(1), image.size(0))  # (W, H)
    else:
        size = image.size  # (W, H)

    is_valid, error_msg = validate_image_size(size, max_size=max_size)
    if not is_valid:
        raise ValueError(error_msg)


class StandardImageTransform:
    """
    Standard image transform with validation and resizing.

    This class provides a complete transform pipeline that includes
    size validation, resizing, and other standard preprocessing steps.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        resize_strategy: str = "maintain_aspect",
        normalize: bool = True,
        augment: bool = False,
        max_size: Tuple[int, int] = (448, 448),
    ):
        """
        Initialize the standard image transform.

        Args:
            target_size: Target image size (width, height)
            resize_strategy: Strategy for resizing
            normalize: Whether to normalize the image
            augment: Whether to apply data augmentation
            max_size: Maximum allowed image size
        """
        self.target_size = target_size
        self.resize_strategy = resize_strategy
        self.normalize = normalize
        self.augment = augment
        self.max_size = max_size

        self.transform = create_standard_transform(
            target_size=target_size,
            resize_strategy=resize_strategy,
            normalize=normalize,
            augment=augment,
        )

    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Any:
        """
        Apply the transform to an image.

        Args:
            image: PIL Image or tensor to transform

        Returns:
            Transformed tensor
        """
        # Validate input size
        validate_transform_input(image, self.max_size)

        # Apply transform
        return self.transform(image)

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
                return validate_image_size(img.size, image_path, self.max_size)
        except Exception as e:
            return False, f"Failed to validate image {image_path}: {e}"
