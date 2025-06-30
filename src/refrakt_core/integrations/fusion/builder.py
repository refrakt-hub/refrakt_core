# refrakt_core/integrations/fusion/builder.py

from typing import Dict, Any
from refrakt_core.integrations.sklearn.wrapper import SklearnWrapper
# from refrakt_core.integrations.cuml.wrapper import CuMLWrapper
from refrakt_core.integrations.fusion.protocols import FusionHead


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

    model_params = dict(params)
    fusion_head_config = model_params.pop("fusion_head", {}) if "fusion_head" in model_params else {}
    
    if fusion_head_config is None:
        fusion_head_config = {}

    if head_type == "sklearn":
        wrapper = SklearnWrapper(model, **model_params)
        if fusion_head_config.get("path"):
            try:
                return SklearnWrapper.load(model, fusion_head_config["path"])
            except (FileNotFoundError, ValueError):
                return wrapper
        return wrapper

    # Future support:
    # if head_type == "cuml":
    #     return CuMLWrapper(model, **params)

    raise ValueError(f"[FusionBuilder] Unsupported fusion head type: {head_type}")
