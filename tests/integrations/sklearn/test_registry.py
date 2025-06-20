from typing import Any
from refrakt_core.integrations.sklearn.registry import load_sklearn_registry

def test_yaml_registry_loads_successfully() -> None:
    """
    Test that the sklearn model registry YAML is parsed and includes required models.
    """
    registry: dict[str, str] = load_sklearn_registry()
    
    assert isinstance(registry, dict), "Registry should be a dictionary"
    assert "random_forest" in registry, "Registry must contain 'random_forest'"
    assert registry["random_forest"] == "sklearn.ensemble.RandomForestClassifier"
