# tests/test_fusion_block.py

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from refrakt_core.integrations.fusion.block import FusionBlock

class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        return self.fc(x)

def test_fusion_block():
    from torch.utils.data import TensorDataset, DataLoader
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    config = {
        "type": "sklearn",
        "model": "random_forest",
        "params": {"n_estimators": 10}
    }

    model = FusionBlock(DummyBackbone(), config)
    model.fit(X_tensor, y_tensor)
    preds = model(X_tensor)

    assert preds.shape == (100,)
