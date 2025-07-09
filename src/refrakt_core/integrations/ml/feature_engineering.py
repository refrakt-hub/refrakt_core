from typing import Any, Dict, List

from sklearn.pipeline import Pipeline  # type: ignore

from refrakt_core.integrations.ml.feature_transformer_registry import (
    FEATURE_TRANSFORMER_REGISTRY,
)


def build_feature_pipeline(steps: List[Dict[str, Any]]) -> Pipeline:
    """
    Build a sklearn Pipeline for feature engineering from a config list.
    Each step is a dict with 'name' and optional 'params'.
    """
    pipeline_steps = []
    for step in steps:
        name = step["name"]
        params = step.get("params", {})
        if name not in FEATURE_TRANSFORMER_REGISTRY:
            raise ValueError(f"Unknown feature transformer: {name}")
        transformer_cls = FEATURE_TRANSFORMER_REGISTRY[name]
        pipeline_steps.append((name, transformer_cls(**params)))
    return Pipeline(pipeline_steps)
