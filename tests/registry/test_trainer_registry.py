import pytest

from refrakt_core.registry.trainer_registry import (
    TRAINER_REGISTRY,
    get_trainer,
    register_trainer,
)


def test_trainer_registration():
    """Test trainer class retrieval"""

    @register_trainer("test_trainer")
    class TestTrainer:
        pass

    trainer_cls = get_trainer("test_trainer")
    assert trainer_cls is TestTrainer


def test_trainer_not_found():
    """Test error for missing trainer"""
    with pytest.raises(ValueError) as excinfo:
        get_trainer("missing_trainer")
    assert "Trainer 'missing_trainer' not found" in str(excinfo.value)
