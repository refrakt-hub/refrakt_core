# refrakt_core/integrations/fusion/builder.py

from typing import Dict, Any
from refrakt_core.integrations.sklearn.wrapper import SklearnWrapper
# from refrakt_core.integrations.cuml.wrapper import CuMLWrapper

def build_fusion_head(cfg: Dict[str, Any]):
    """
    Construct a fusion head from a config dictionary.
    """
    head_type = cfg["type"]
    model = cfg["model"]
    params = cfg.get("params", {})

    if head_type == "sklearn":
        return SklearnWrapper(model, **params)

    # if head_type == "cuml":
    #     return CuMLWrapper(model, **params)

    raise ValueError(f"Unsupported fusion head type: {head_type}")
