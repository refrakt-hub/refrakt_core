"""
Tests for the improved logging configuration system.

This module contains comprehensive tests for the improved logging configuration
including smoke tests, sanity checks, and unit tests.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from refrakt_core.logging_config import (
    RefraktLoggingManager,
    configure_logger,
    get_config,
    get_global_logger,
    get_logger,
    get_logging_manager,
    list_loggers,
    remove_logger,
    reset_global_logger,
    set_global_logger,
    temporary_logger,
    update_config,
)


# Smoke Tests
def test_logging_manager_singleton_smoke():
    """Smoke test: Verify singleton pattern works correctly."""
    manager1 = RefraktLoggingManager()
    manager2 = RefraktLoggingManager()
    assert manager1 is manager2
    assert id(manager1) == id(manager2)


def test_configure_logger_basic_smoke():
    """Smoke test: Configure a basic logger."""
    manager = RefraktLoggingManager()
    logger = manager.configure_logger("test_logger")
    assert logger is not None
    assert logger.name == "refrakt:test_logger"
    assert logger.level == logging.INFO


def test_get_logger_existing_smoke():
    """Smoke test: Get an existing logger."""
    manager = RefraktLoggingManager()
    original_logger = manager.configure_logger("test_logger")
    retrieved_logger = manager.get_logger("test_logger")
    assert retrieved_logger is original_logger


# Sanity Tests
def test_configure_logger_with_options_sanity():
    """Sanity test: Configure logger with custom options."""
    manager = RefraktLoggingManager()
    logger = manager.configure_logger(
        "test_logger",
        log_dir="./test_logs",
        console=True,
        debug=True,
        log_types=["tensorboard"],
    )
    assert logger is not None
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) > 0


def test_update_config_sanity():
    """Sanity test: Update logger configuration."""
    manager = RefraktLoggingManager()
    manager.configure_logger("test_logger")
    manager.update_config("test_logger", debug=True, console=False)
    config = manager.get_config("test_logger")
    assert config["debug"] is True
    assert config["console"] is False


def test_remove_logger_sanity():
    """Sanity test: Remove a logger."""
    manager = RefraktLoggingManager()
    manager.configure_logger("test_logger")
    loggers = manager.list_loggers()
    assert "test_logger" in loggers
    manager.remove_logger("test_logger")
    loggers = manager.list_loggers()
    assert "test_logger" not in loggers


# Unit Tests
def test_logger_file_output_unit():
    """Unit test: Test logger file output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = RefraktLoggingManager()
        logger = manager.configure_logger(
            "test_logger", log_dir=temp_dir, console=False
        )
        test_message = "Test log message"
        logger.info(test_message)
        log_files = list(Path(temp_dir).glob("test_logger/*.log"))
        assert len(log_files) == 1


def test_list_loggers_unit():
    """Unit test: List all loggers."""
    manager = RefraktLoggingManager()
    manager.configure_logger("logger1")
    manager.configure_logger("logger2")
    loggers = manager.list_loggers()
    assert "logger1" in loggers
    assert "logger2" in loggers


def test_temporary_logger_context_unit():
    """Unit test: Test temporary logger context manager."""
    manager = RefraktLoggingManager()
    manager.configure_logger("test_logger", debug=False)
    with manager.temporary_logger("test_logger", debug=True) as temp_logger:
        config = manager.get_config("test_logger")
        assert config["debug"] is True
        temp_logger.info("Test message")
    config = manager.get_config("test_logger")
    assert config["debug"] is False


def test_get_config_nonexistent_logger_unit():
    """Unit test: Get config for nonexistent logger."""
    manager = RefraktLoggingManager()
    config = manager.get_config("nonexistent_logger")
    assert isinstance(config, dict)


def test_remove_nonexistent_logger_unit():
    """Unit test: Remove nonexistent logger (should not raise)."""
    manager = RefraktLoggingManager()
    manager.remove_logger("nonexistent_logger")
    # Should not raise
