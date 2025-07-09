"""
Comprehensive tests for sklearn trainer module.
"""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from refrakt_core.integrations.cpu.wrapper import SklearnWrapper
from refrakt_core.integrations.fusion.trainer import FusionTrainer
from refrakt_core.schema.model_output import ModelOutput


class DummyBackbone(nn.Module):
    """Dummy backbone for testing."""

    def __init__(self, feature_dim: int = 10):
        super().__init__()
        self.fc = nn.Linear(20, feature_dim)

    def forward(self, x):
        embeddings = self.fc(x)
        return ModelOutput(embeddings=embeddings)


def create_dummy_data(n_samples: int = 100, n_features: int = 20, n_classes: int = 2):
    """Create dummy data for testing."""
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return X, y


def create_dummy_dataloaders(n_samples: int = 100, batch_size: int = 32):
    """Create dummy dataloaders for testing."""
    X, y = create_dummy_data(n_samples)
    train_dataset = TensorDataset(X, y)
    val_dataset = TensorDataset(X, y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def test_trainer_initialization_smoke():
    """Smoke test: Trainer initializes correctly."""
    train_loader, val_loader = create_dummy_dataloaders()
    backbone = DummyBackbone()
    fusion_head = SklearnWrapper("random_forest", n_estimators=5)

    trainer = FusionTrainer(
        model=backbone,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    assert trainer.model == backbone
    assert trainer.fusion_head == fusion_head
    assert str(trainer.device) == "cpu"


def test_trainer_training_smoke():
    """Smoke test: Trainer can train and return results."""
    train_loader, val_loader = create_dummy_dataloaders()
    backbone = DummyBackbone()
    fusion_head = SklearnWrapper("random_forest", n_estimators=5)

    trainer = FusionTrainer(
        model=backbone,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    results = trainer.train(num_epochs=1)

    assert isinstance(results, dict)
    assert "fusion_accuracy" in results
    assert isinstance(results["fusion_accuracy"], float)
    assert 0.0 <= results["fusion_accuracy"] <= 1.0


def test_trainer_evaluation_smoke():
    """Smoke test: Trainer can evaluate and return accuracy."""
    train_loader, val_loader = create_dummy_dataloaders()
    backbone = DummyBackbone()
    fusion_head = SklearnWrapper("random_forest", n_estimators=5)

    trainer = FusionTrainer(
        model=backbone,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    # Train first
    trainer.train(num_epochs=1)

    # Then evaluate
    accuracy = trainer.evaluate()

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_feature_extraction_sanity():
    """Sanity test: Verify feature extraction works correctly."""
    train_loader, val_loader = create_dummy_dataloaders()
    backbone = DummyBackbone(feature_dim=5)
    fusion_head = SklearnWrapper("random_forest", n_estimators=5)

    trainer = FusionTrainer(
        model=backbone,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    # Test feature extraction
    X_train, y_train = trainer._extract_features_and_labels(train_loader)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape[0] == y_train.shape[0]  # Same number of samples
    assert X_train.shape[1] == 5  # Feature dimension from backbone


def test_batch_unpacking_sanity():
    """Sanity test: Verify batch unpacking handles different formats."""
    trainer = FusionTrainer(
        model=DummyBackbone(),
        fusion_head=SklearnWrapper("random_forest", n_estimators=5),
        train_loader=create_dummy_dataloaders()[0],
        val_loader=create_dummy_dataloaders()[1],
        device="cpu",
    )

    # Test tuple format (img1, img2, label) - SimCLR style
    batch_tuple = (torch.randn(32, 20), torch.randn(32, 20), torch.randint(0, 2, (32,)))
    x, y = trainer._unpack_batch(batch_tuple)
    assert x.shape == (32, 20)
    assert y.shape == (32,)

    # Test tuple format (img, label)
    batch_simple = (torch.randn(32, 20), torch.randint(0, 2, (32,)))
    x, y = trainer._unpack_batch(batch_simple)
    assert x.shape == (32, 20)
    assert y.shape == (32,)

    # Test dict format
    batch_dict = {"input": torch.randn(32, 20), "target": torch.randint(0, 2, (32,))}
    x, y = trainer._unpack_batch(batch_dict)
    assert x.shape == (32, 20)
    assert y.shape == (32,)


def test_artifact_dumper_integration_sanity():
    """Sanity test: Verify artifact dumper integration."""
    train_loader, val_loader = create_dummy_dataloaders()
    backbone = DummyBackbone()
    fusion_head = SklearnWrapper("random_forest", n_estimators=5)

    # Mock artifact dumper
    mock_dumper = Mock()
    mock_dumper.log_scalar_dict = Mock()

    trainer = FusionTrainer(
        model=backbone,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        artifact_dumper=mock_dumper,
    )

    # Train and evaluate
    trainer.train(num_epochs=1)
    accuracy = trainer.evaluate()

    # Check that artifact dumper was called
    assert mock_dumper.log_scalar_dict.call_count == 2
    call_args = mock_dumper.log_scalar_dict.call_args
    assert call_args[0][0] == {"fusion_accuracy": accuracy}
    assert call_args[1]["prefix"] == "val"


def test_model_output_validation_unit():
    """Unit test: Verify ModelOutput validation."""
    train_loader, val_loader = create_dummy_dataloaders()

    # Create backbone that doesn't return ModelOutput with embeddings
    class InvalidBackbone(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 10)  # No ModelOutput

    backbone = InvalidBackbone()
    fusion_head = SklearnWrapper("random_forest", n_estimators=5)

    trainer = FusionTrainer(
        model=backbone,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    with pytest.raises(
        ValueError, match="Backbone must return `ModelOutput` with `embeddings`"
    ):
        trainer.train(num_epochs=1)


def test_model_output_no_embeddings_unit():
    """Unit test: Verify error when ModelOutput has no embeddings."""
    train_loader, val_loader = create_dummy_dataloaders()

    class NoEmbeddingsBackbone(nn.Module):
        def forward(self, x):
            return ModelOutput(logits=torch.randn(x.shape[0], 2))  # No embeddings

    backbone = NoEmbeddingsBackbone()
    fusion_head = SklearnWrapper("random_forest", n_estimators=5)

    trainer = FusionTrainer(
        model=backbone,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    with pytest.raises(
        ValueError, match="Backbone must return `ModelOutput` with `embeddings`"
    ):
        trainer.train(num_epochs=1)


def test_invalid_batch_format_unit():
    """Unit test: Verify error handling for invalid batch formats."""
    trainer = FusionTrainer(
        model=DummyBackbone(),
        fusion_head=SklearnWrapper("random_forest", n_estimators=5),
        train_loader=create_dummy_dataloaders()[0],
        val_loader=create_dummy_dataloaders()[1],
        device="cpu",
    )

    # Test invalid batch format
    invalid_batch = "invalid_batch"

    with pytest.raises(TypeError, match="Unsupported batch format"):
        trainer._unpack_batch(invalid_batch)


def test_device_handling_unit():
    """Unit test: Verify device handling."""
    train_loader, val_loader = create_dummy_dataloaders()
    backbone = DummyBackbone()
    fusion_head = SklearnWrapper("random_forest", n_estimators=5)

    # Test with CPU device
    trainer_cpu = FusionTrainer(
        model=backbone,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    # Test with CUDA device (if available)
    if torch.cuda.is_available():
        trainer_cuda = FusionTrainer(
            model=backbone,
            fusion_head=fusion_head,
            train_loader=train_loader,
            val_loader=val_loader,
            device="cuda",
        )
        assert str(trainer_cuda.device) == "cuda"


def test_extra_params_handling_unit():
    """Unit test: Verify extra parameters handling."""
    train_loader, val_loader = create_dummy_dataloaders()
    backbone = DummyBackbone()
    fusion_head = SklearnWrapper("random_forest", n_estimators=5)

    trainer = FusionTrainer(
        model=backbone,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        extra_param1="value1",
        extra_param2="value2",
    )

    assert trainer.extra_params["extra_param1"] == "value1"
    assert trainer.extra_params["extra_param2"] == "value2"


def test_global_step_tracking_unit():
    """Unit test: Verify global step tracking."""
    train_loader, val_loader = create_dummy_dataloaders()
    backbone = DummyBackbone()
    fusion_head = SklearnWrapper("random_forest", n_estimators=5)

    trainer = FusionTrainer(
        model=backbone,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    assert trainer.global_step == 0

    # Train should not increment global step (by current implementation)
    trainer.train(num_epochs=1)
    assert trainer.global_step == 0


def test_single_sample_handling_unit():
    """Unit test: Verify handling of single sample batches."""
    # Create dataloaders with batch_size=1
    train_loader, val_loader = create_dummy_dataloaders(n_samples=10, batch_size=1)
    backbone = DummyBackbone()
    fusion_head = SklearnWrapper("random_forest", n_estimators=5)

    trainer = FusionTrainer(
        model=backbone,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    # Should not raise any errors
    results = trainer.train(num_epochs=1)
    assert "fusion_accuracy" in results
