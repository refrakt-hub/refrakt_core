import importlib
import yaml
from pathlib import Path

FEATURE_TRANSFORMER_REGISTRY = {}

def load_feature_transformer_registry(yaml_path=None):
    # Try local ml/ directory first
    if yaml_path is None:
        local_path = Path(__file__).parent / "feature_transformer_registry.yaml"
        registry_path = Path(__file__).parent.parent / "registry/feature_transformer_registry.yaml"
        if local_path.exists():
            yaml_path = local_path
        elif registry_path.exists():
            yaml_path = registry_path
        else:
            raise FileNotFoundError(f"feature_transformer_registry.yaml not found in {local_path} or {registry_path}")
    with open(yaml_path, "r") as f:
        mapping = yaml.safe_load(f)
    for key, import_path in mapping.items():
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        FEATURE_TRANSFORMER_REGISTRY[key] = cls

# Load at import
load_feature_transformer_registry() 