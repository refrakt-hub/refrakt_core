"""
Tests for the hyperparameter override system.

This module contains comprehensive tests for the hyperparameter override system
including smoke tests, sanity checks, and unit tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from refrakt_core.hooks.hyperparameter_override import (
    _convert_value,
    _set_nested_value,
    apply_overrides,
    create_override_help,
    extract_overrides_from_args,
    get_parameter_value,
    load_config_with_overrides,
    merge_configs,
    parse_override_arg,
    save_config_with_overrides,
    set_parameter_value,
    validate_config_structure,
    validate_overrides,
)


# Smoke Tests
def test_parse_override_basic_smoke():
    """Smoke test: Parse basic override argument."""
    path, value = parse_override_arg("config.model.name=ResNet")

    assert path == "config.model.name"
    assert value == "ResNet"


def test_convert_value_smoke():
    """Smoke test: Convert different value types."""
    # String
    assert _convert_value("hello") == "hello"

    # Integer
    assert _convert_value("42") == 42

    # Float
    assert _convert_value("3.14") == 3.14

    # Boolean
    assert _convert_value("true") is True
    assert _convert_value("false") is False


def test_apply_overrides_basic_smoke():
    """Smoke test: Apply basic overrides."""
    from omegaconf import OmegaConf
    config = OmegaConf.create({"model": {"name": "default"}, "epochs": 10})
    overrides = ["model.name=ResNet", "epochs=20"]

    result = apply_overrides(config, overrides)

    assert result["model"]["name"] == "ResNet"
    assert result["epochs"] == 20


def test_set_nested_value_smoke():
    """Smoke test: Set nested value."""
    config = OmegaConf.create({"model": {"name": "default"}})

    _set_nested_value(config, "model.name", "ResNet")

    assert config["model"]["name"] == "ResNet"


def test_get_parameter_value_smoke():
    """Smoke test: Get parameter value."""
    config = OmegaConf.create({"model": {"name": "ResNet", "params": {"lr": 0.001}}})

    value = get_parameter_value(config, "model.name")
    assert value == "ResNet"

    value = get_parameter_value(config, "model.params.lr")
    assert value == 0.001


def test_set_parameter_value_smoke():
    """Smoke test: Set parameter value."""
    config = OmegaConf.create({"model": {"name": "default"}})

    set_parameter_value(config, "model.name", "ResNet")

    assert config["model"]["name"] == "ResNet"


# Sanity Tests
def test_parse_override_sanity():
    """Sanity test: Parse different override formats."""
    # Numeric values
    path, value = parse_override_arg("config.epochs=100")
    assert path == "config.epochs"
    assert value == 100

    # Float values
    path, value = parse_override_arg("config.lr=0.001")
    assert path == "config.lr"
    assert value == 0.001

    # Boolean values
    path, value = parse_override_arg("config.debug=true")
    assert path == "config.debug"
    assert value is True


def test_apply_overrides_sanity():
    """Sanity test: Apply nested overrides."""
    from omegaconf import OmegaConf
    config = OmegaConf.create({"model": {"params": {"lr": 0.001, "batch_size": 32}}})
    overrides = ["model.params.lr=0.0005", "model.params.batch_size=64"]

    result = apply_overrides(config, overrides)

    assert result["model"]["params"]["lr"] == 0.0005
    assert result["model"]["params"]["batch_size"] == 64


def test_validate_overrides_sanity():
    """Sanity test: Validate valid overrides."""
    overrides = [
        "config.model.name=ResNet",
        "config.optimizer.lr=0.001",
        "config.epochs=100",
    ]

    errors = validate_overrides(overrides)

    assert len(errors) == 0


def test_extract_overrides_sanity():
    """Sanity test: Extract overrides from arguments."""
    args = [
        "python",
        "train.py",
        "config.model.name=ResNet",
        "config.epochs=100",
        "--device",
        "cuda",
    ]

    overrides, remaining = extract_overrides_from_args(args)

    assert overrides == ["config.model.name=ResNet", "config.epochs=100"]
    assert remaining == ["python", "train.py", "--device", "cuda"]


def test_merge_configs_sanity():
    """Sanity test: Merge configurations."""
    base_config = OmegaConf.create({"model": {"name": "default"}, "epochs": 10})
    override_config = OmegaConf.create({"model": {"name": "ResNet"}, "lr": 0.001})

    result = merge_configs(base_config, override_config)

    assert result["model"]["name"] == "ResNet"
    assert result["epochs"] == 10
    assert result["lr"] == 0.001


# Unit Tests
def test_parse_override_invalid_format():
    """Unit test: Test invalid override format."""
    with pytest.raises(ValueError, match="Invalid override format"):
        parse_override_arg("config.model.name")


def test_parse_override_with_spaces():
    """Unit test: Test override with spaces."""
    path, value = parse_override_arg("  config.model.name = ResNet  ")

    assert path == "config.model.name"
    assert value == "ResNet"


def test_apply_overrides_create_new_paths():
    """Unit test: Create new paths when they don't exist."""
    from omegaconf import OmegaConf
    config = OmegaConf.create({"existing": "value"})
    overrides = ["new.path=value", "another.nested.path=42"]

    result = apply_overrides(config, overrides)

    assert result["new"]["path"] == "value"
    assert result["another"]["nested"]["path"] == 42
    assert result["existing"] == "value"  # Original preserved


def test_apply_overrides_invalid_override():
    """Unit test: Test invalid override handling."""
    from omegaconf import OmegaConf
    config = OmegaConf.create({"model": {"name": "default"}})
    overrides = ["invalid.override"]

    with pytest.raises(ValueError, match="Failed to apply override"):
        apply_overrides(config, overrides)


def test_set_nested_value_create_path():
    """Unit test: Create path when it doesn't exist."""
    config = OmegaConf.create({"existing": "value"})

    _set_nested_value(config, "new.nested.path", "value")

    assert config["new"]["nested"]["path"] == "value"
    assert config["existing"] == "value"  # Original preserved


def test_validate_overrides_invalid_format():
    """Unit test: Validate invalid override format."""
    overrides = ["config.model.name=ResNet", "invalid.override", "config.epochs=100"]

    errors = validate_overrides(overrides)

    assert len(errors) == 1
    assert "invalid.override" in errors[0]


def test_get_parameter_value_nested_missing():
    """Unit test: Get parameter value with missing nested path."""
    config = OmegaConf.create({"model": {"name": "ResNet"}})

    # Should return None for missing path
    value = get_parameter_value(config, "model.params.lr")
    assert value is None


def test_set_parameter_value_invalid_path():
    """Unit test: Set parameter value with invalid path."""
    config = OmegaConf.create({"model": {"name": "default"}})

    # Should handle invalid path gracefully
    set_parameter_value(config, "invalid.path", "value")

    assert config["invalid"]["path"] == "value"


def test_create_override_help():
    """Unit test: Create override help text."""
    help_text = create_override_help()

    assert isinstance(help_text, str)
    assert "override" in help_text.lower()


def test_load_config_with_overrides():
    """Unit test: Load config with overrides."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("model:\n  name: default\nepochs: 10\n")
        config_path = f.name

    try:
        overrides = ["model.name=ResNet", "epochs=20"]
        result = load_config_with_overrides(config_path, overrides)

        assert result["model"]["name"] == "ResNet"
        assert result["epochs"] == 20
    finally:
        Path(config_path).unlink()


def test_save_config_with_overrides():
    """Unit test: Save config with overrides."""
    from omegaconf import OmegaConf

    config = OmegaConf.create({"model": {"name": "ResNet"}, "epochs": 20})
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        config_path = f.name
    try:
        save_config_with_overrides(config, config_path)
        # Verify file was created
        assert Path(config_path).exists()
    finally:
        Path(config_path).unlink()


def test_validate_config_structure():
    """Unit test: Validate config structure."""
    config = OmegaConf.create({"model": {"name": "ResNet"}, "epochs": 20})
    required_paths = ["model.name", "epochs"]
    # Should not raise any exceptions for valid config
    errors = validate_config_structure(config, required_paths)
    assert errors == []
    # Should handle invalid config gracefully - test with missing paths
    invalid_config = OmegaConf.create({"model": {"name": "ResNet"}})  # Missing "epochs"
    errors = validate_config_structure(invalid_config, required_paths)
    assert len(errors) == 1  # Should have 1 error for missing "epochs"
    assert "epochs" in errors[0]
