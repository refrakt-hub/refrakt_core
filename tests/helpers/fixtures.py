# tests/trainer/conftest.py
import pytest
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.data = torch.randn(10, 3, 32, 32)
        self.labels = torch.randint(0, 2, (10,))
        # Accept and ignore all extra args/kwargs
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ContrastiveDataset(Dataset):
    def __init__(self, size=32, num_samples=100):
        self.data = torch.randn(num_samples, 3, size, size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx].flip(-1)  # Simple augmentation


@pytest.fixture
def dummy_dataset():
    return DummyDataset()


@pytest.fixture
def contrastive_dataset():
    return ContrastiveDataset()


@pytest.fixture
def train_loader(dummy_dataset):
    return DataLoader(dummy_dataset, batch_size=8)


@pytest.fixture
def val_loader(dummy_dataset):
    return DataLoader(dummy_dataset, batch_size=8)


@pytest.fixture
def contrastive_loader(contrastive_dataset):
    return DataLoader(contrastive_dataset, batch_size=8)


@pytest.fixture
def dummy_cfg():
    # Return a minimal valid DictConfig for use in tests
    return DictConfig(
        {
            "model": {"name": "dummy", "params": {}},
            "dataset": {"name": "dummy", "params": {}},
            "dataloader": {"batch_size": 1},
            "trainer": {
                "name": "dummy_trainer",
                "params": {"save_dir": "./checkpoints"},
            },
            "loss": {"name": "dummy_loss", "params": {}},
        }
    )


# Simple models for testing
class SimpleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32 * 32 * 3, 128), nn.ReLU())
        self.decoder = nn.Linear(128, 32 * 32 * 3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.decoder(self.encoder(x))


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class DummyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Add a dummy parameter so parameters() is not empty
        self.dummy_param = nn.Parameter(torch.zeros(1))
        # Accept and ignore all extra args/kwargs
        pass

    def forward(self, x=None, *args, **kwargs):
        return x

    def to(self, device=None, *args, **kwargs):
        return self
