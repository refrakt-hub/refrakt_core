from typing import Any, Dict, Union, cast

from refrakt_core.integrations.cpu.wrapper import SklearnWrapper
from refrakt_core.integrations.gpu.wrapper import CuMLWrapper

ML_WRAPPERS = {
    "sklearn": SklearnWrapper,
    "cuml": CuMLWrapper,
}


def build_ml_model(cfg: Dict[str, Any]) -> Union[SklearnWrapper, CuMLWrapper]:
    """
    Build a pure-ML model (sklearn/cuml) from config dict.
    cfg should have 'backend' (or 'type'), 'name' (or 'model'), and 'params'.
    """
    ml_type = cfg.get("backend") or cfg.get("type") or "sklearn"  # Default to sklearn
    model = cfg.get("name") or cfg.get("model")
    if model is None:
        raise ValueError("Model name must be specified in config")
    params = cfg.get("params", {})
    if ml_type not in ML_WRAPPERS:
        raise ValueError(f"Unknown ML backend: {ml_type}")
    wrapper_cls = ML_WRAPPERS[ml_type]
    return cast(Union[SklearnWrapper, CuMLWrapper], wrapper_cls(model, **params))
