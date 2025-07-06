# refrakt_core/integrations/fusion/builder.py

from typing import Any, Dict

from refrakt_core.integrations.gpu.wrapper import CuMLWrapper
from refrakt_core.integrations.fusion.protocols import FusionHead
from refrakt_core.integrations.cpu.wrapper import SklearnWrapper
from refrakt_core.integrations.fusion.utils import (
    create_sklearn_wrapper,
    create_cuml_wrapper,
    try_load_wrapper_from_path,
    validate_head_type
)


def build_fusion_head(cfg: Dict[str, Any]) -> FusionHead:
    """
    Construct a fusion head from a config dictionary.

    Args:
        cfg (Dict[str, Any]): Fusion config dictionary. Should contain:
            - type: "sklearn" or other ML framework
            - model: key or class path for model (e.g., "random_forest" or "sklearn.ensemble.RandomForestClassifier")
            - params: hyperparameters for the fusion head and wrapper configuration

    Returns:
        FusionHead: An instance that implements the FusionHead protocol.

    Raises:
        ValueError: If the fusion head type is unsupported.
    """
    head_type = cfg["type"].lower()
    model = cfg["model"]
    params = cfg.get("params", {})

    validate_head_type(head_type)

    if head_type == "sklearn":
        wrapper = create_sklearn_wrapper(model, params)
        fusion_head_config = params.get("fusion_head", {})
        return try_load_wrapper_from_path(wrapper, model, fusion_head_config)

    if head_type == "cuml":
        wrapper = create_cuml_wrapper(model, params)
        fusion_head_config = params.get("fusion_head", {})
        return try_load_wrapper_from_path(wrapper, model, fusion_head_config)

    # This should never be reached due to validate_head_type, but kept for safety
    raise ValueError(f"[FusionBuilder] Unsupported fusion head type: {head_type}")
