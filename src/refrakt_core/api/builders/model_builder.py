"""# model_builder.py"""
from typing import Any, Dict
from omegaconf import OmegaConf

from refrakt_core.registry.model_registry import get_model
from refrakt_core.registry.wrapper_registry import get_wrapper


# refrakt_core/builders/build_model.py

from typing import Any, Dict
from omegaconf import OmegaConf

from refrakt_core.registry.model_registry import get_model
from refrakt_core.registry.wrapper_registry import get_wrapper

def build_model(cfg: OmegaConf, modules: Dict, device: str) -> Any:
    """
    Builds a model, optionally using a registered wrapper.
    """
    print("Building model...")
    model_params = cfg.model.params or {}
    model_name = cfg.model.name
    wrapper_name = cfg.model.get("wrapper", None)

    if wrapper_name:
        print(f"Using wrapper: {wrapper_name} for model: {model_name}")
        wrapper_cls = get_wrapper(wrapper_name)
        model = wrapper_cls(model_name=model_name, model_params=model_params).to(device)
    else:
        model = modules["get_model"](model_name, **model_params).to(device)

    print(f"Model: {model_name} with params: {model_params}")
    return model
