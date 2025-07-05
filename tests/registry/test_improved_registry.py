"""
Tests for the improved registry system.

This module contains comprehensive tests for the improved registry system
including smoke tests, sanity checks, and unit tests.
"""

import pytest
import threading
import time
from unittest.mock import Mock

from refrakt_core.registry.improved_registry import (
    SafeRegistry,
    get_registry,
    register_component,
    get_component,
    list_components,
    clear_registry,
    register_model,
    get_model,
    register_dataset,
    get_dataset,
    register_loss,
    get_loss,
    register_trainer,
    get_trainer,
    register_transform,
    get_transform
)

# Smoke Tests
def test_safe_registry_singleton_smoke():
    """Smoke test: Verify singleton pattern works correctly."""
    registry1 = SafeRegistry()
    registry2 = SafeRegistry()
    assert registry1 is registry2
    assert id(registry1) == id(registry2)


def test_safe_registry_thread_safety_smoke():
    """Smoke test: Verify thread safety of registry operations."""
    registry = SafeRegistry()
    registry.clear("test")
    def worker(worker_id: int):
        for i in range(5):
            registry.register("test", f"component_{worker_id}_{i}", f"value_{worker_id}_{i}")
            time.sleep(0.001)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    components = registry.list_components("test")
    assert len(components) == 10  # 2 threads * 5 components each


def test_register_and_get_component_smoke():
    """Smoke test: Register and retrieve a component."""
    registry = SafeRegistry()
    registry.clear("test_registry")
    test_component = {"name": "test", "value": 42}
    registry.register("test_registry", "test_component", test_component)
    retrieved = registry.get("test_registry", "test_component")
    assert retrieved == test_component

# Sanity Tests
def test_get_nonexistent_component_sanity():
    """Sanity test: Test getting a non-existent component."""
    registry = SafeRegistry()
    registry.clear("test_registry")
    with pytest.raises(ValueError, match="Component 'nonexistent' not found"):
        registry.get("test_registry", "nonexistent")


def test_list_components_sanity():
    """Sanity test: Test listing components in a registry."""
    registry = SafeRegistry()
    registry.clear("test_registry")
    registry.register("test_registry", "comp1", "value1")
    registry.register("test_registry", "comp2", "value2")
    components = registry.list_components("test_registry")
    assert "comp1" in components
    assert "comp2" in components
    assert len(components) == 2


def test_clear_registry_sanity():
    """Sanity test: Test clearing a registry."""
    registry = SafeRegistry()
    registry.register("test_registry", "comp1", "value1")
    registry.clear("test_registry")
    components = registry.list_components("test_registry")
    assert len(components) == 0

# Unit Tests
def test_register_component_decorator_unit():
    """Unit test: Register component decorator."""
    @register_component("test_registry", "test_component")
    class TestComponent:
        def __init__(self):
            self.value = 123
    instance = TestComponent()
    result = get_component("test_registry", "test_component")
    assert result is TestComponent
    assert isinstance(instance, TestComponent)


def test_register_model_unit():
    """Unit test: Register model decorator."""
    @register_model("test_model")
    class TestModel:
        pass
    assert get_model("test_model") is TestModel


def test_register_dataset_unit():
    """Unit test: Register dataset decorator."""
    @register_dataset("test_dataset")
    class TestDataset:
        pass
    assert get_dataset("test_dataset") is TestDataset


def test_import_callback_unit():
    """Unit test: Test import callback functionality."""
    registry = SafeRegistry()
    # Clear all registries to ensure fresh state
    registry.clear()
    callback_called = []
    def mock_callback():
        callback_called.append(True)
    registry.register_import_callback("test_registry", mock_callback)
    # Access the registry to trigger the callback
    registry.list_components("test_registry")
    assert callback_called


def test_temporary_registry_context_unit():
    """Unit test: Test temporary registry context manager."""
    registry = SafeRegistry()
    registry.register("test_registry", "original", "original_value")
    with registry.temporary_registry("test_registry"):
        registry.register("test_registry", "temp", "temp_value")
        components = registry.list_components("test_registry")
        assert "original" in components
        assert "temp" in components
    components = registry.list_components("test_registry")
    assert "original" in components
    assert "temp" not in components 