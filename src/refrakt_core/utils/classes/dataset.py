"""Custom image classification dataset loader for structured folders."""

from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from refrakt_core.utils.methods import find_classes


class CreateDataset(Dataset[Any]):
    """
    A custom dataset for loading images from a directory structured as:
    root/class_name/image.jpg

    Args:
        target_dir (str): Path to the root dataset directory.
        transform (Optional[Callable]): Transformations to apply to each image.
    """

    def __init__(
        self, target_dir: str, transform: Optional[Callable[[Any], Tensor]] = None
    ) -> None:
        self.paths = list(Path(target_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_dir)

    def load_image(self, index: int) -> Image.Image:
        """
        Loads an image at the specified index.

        Args:
            index (int): Index of the image to load.

        Returns:
            Image.Image: The loaded image.
        """
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Dataset length.
        """
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Returns the transformed image tensor and its corresponding class index.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[Tensor, int]: (transformed image tensor, class index)
        """
        image = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            image_tensor = self.transform(image)
            return image_tensor, class_idx
        raise TypeError("Transform must be provided to return a Tensor.")
