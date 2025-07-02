# tests/trainer/conftest.py
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, size=32, num_samples=100):
        self.data = torch.randn(num_samples, 3, size, size)
        self.targets = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


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
