# refrakt_core/builders/build_model.py

from typing import Any, Dict
from omegaconf import OmegaConf

from refrakt_core.registry.wrapper_registry import get_wrapper
from refrakt_core.wrappers.schema.default_model import DefaultModelWrapper


def build_model(cfg: OmegaConf, modules: Dict[str, Any], device: str) -> Any:
    """
    Builds a model using a registered wrapper if available.
    Falls back to DefaultModelWrapper if wrapper fails or not specified.
    Ensures output is always ModelOutput-compliant.
    """
    print("Building model...")

    model_params = cfg.model.get("params", {}) or {}
    model_name = cfg.model.name
    wrapper_name = cfg.model.get("wrapper", None)

    if wrapper_name:
        print(f"[INFO] Attempting to use wrapper '{wrapper_name}' for model '{model_name}'")
        try:
            wrapper_cls = get_wrapper(wrapper_name)
            model = wrapper_cls(model_name=model_name, model_params=model_params).to(device)
            print(f"[SUCCESS] Wrapped model '{model_name}' with '{wrapper_name}'")
        except Exception as e:
            print(f"[WARN] Wrapper '{wrapper_name}' failed: {e}")
            print(f"[FALLBACK] Using DefaultModelWrapper instead.")
            model = DefaultModelWrapper(
                model_name=model_name,
                model_params=model_params,
                modules=modules
            ).to(device)
    else:
        print(f"[INFO] No wrapper specified. Using DefaultModelWrapper for model '{model_name}'")
        model = DefaultModelWrapper(
            model_name=model_name,
            model_params=model_params,
            modules=modules
        ).to(device)

    print(f"[FINALIZED] Model: {model_name} with params: {model_params}")
    return model