"""
Contains a set of dataset class for family of models. 
Available dataset classes are: 
- ContrastiveDataset
- SuperResolutionDataset
"""

import os
from pathlib import Path

from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from refrakt_core.registry.dataset_registry import register_dataset


@register_dataset("contrastive")
class ContrastiveDataset(Dataset):
    """
    A wrapper that sets up a dataset class for contrastive learning methods, 
    like SimCLR and DINO. Further models to be implemented in the future. 
    """
    def __init__(self, base_dataset, transform=None, train=None):
        self.base_dataset = base_dataset
        self.transform = transform

        if self.transform:
            self.transform.transforms = [
                t for t in self.transform.transforms if not isinstance(t, nn.Flatten)
            ]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        # Handle different dataset formats
        if isinstance(item, tuple) and len(item) >= 2:
            x = item[0]  # Assume first element is image
        else:
            x = item  # Assume single element is image

        # Apply transform if available
        if self.transform:
            view1 = self.transform(x)
            view2 = self.transform(x)
            return view1, view2
        # Return original image twice if no transform
        return x, x


@register_dataset("super_resolution")
class SuperResolutionDataset(Dataset):
    """
    A dataset class for super-resolution based training. 
    """
    def __init__(self, lr_dir, hr_dir, transform=None, train=None):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.filenames = sorted(os.listdir(self.lr_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr = Image.open(self.lr_dir / fname).convert("RGB")
        hr = Image.open(self.hr_dir / fname).convert("RGB")

        if self.transform:
            lr, hr = self.transform(lr, hr)

        return {"lr": lr, "hr": hr}
