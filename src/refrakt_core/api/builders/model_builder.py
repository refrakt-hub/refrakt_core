from typing import Any, Dict
from omegaconf import OmegaConf
import inspect

from refrakt_core.registry.wrapper_registry import load_wrapper
from refrakt_core.wrappers.schema.default_model import DefaultModelWrapper


def build_model(cfg: OmegaConf, modules: Dict[str, Any], device: str) -> Any:
    import refrakt_core.models

    model_params = cfg.model.get("params", {}) or {}
    model_name = cfg.model.name
    wrapper_name = cfg.model.get("wrapper", None)

    try:
        get_model_fn = modules.get("get_model")
        if get_model_fn is None:
            raise ValueError("[ERROR] get_model function not found in modules registry.")

        get_wrapper_fn = modules.get("get_wrapper")
        if wrapper_name and get_wrapper_fn is None:
            raise ValueError("[ERROR] get_wrapper function not found in modules registry.")

        # Convert DictConfig to regular dict if needed
        model_params_dict = dict(model_params) if hasattr(model_params, "items") else model_params

        # Step 1: Instantiate base model
        model_cls = get_model_fn(model_name)
        raw_model = model_cls(**model_params_dict).to(device)

        # Step 2: Wrap model (if wrapper is specified)
        if wrapper_name:
            print(f"[INFO] Attempting to use wrapper '{wrapper_name}' for model '{model_name}'")

            wrapper_cls = get_wrapper_fn(wrapper_name)
            sig = inspect.signature(wrapper_cls.__init__)
            valid_params = set(sig.parameters.keys()) - {"self", "model", "args", "kwargs"}

            wrapper_args = {
                k: v for k, v in model_params_dict.items() if k in valid_params
            }

            model = wrapper_cls(model=raw_model, **wrapper_args).to(device)
            print(f"[SUCCESS] Wrapped model '{model_name}' with '{wrapper_name}'")
        else:
            print(f"[INFO] No wrapper specified. Using DefaultModelWrapper for model '{model_name}'")
            model = DefaultModelWrapper(
                model_name=model_name,
                model_params=model_params,
                modules=modules
            ).to(device)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FALLBACK] Using DefaultModelWrapper due to error: {e}")
        model = DefaultModelWrapper(
            model_name=model_name,
            model_params=model_params,
            modules=modules
        ).to(device)

    print(f"[FINALIZED] Model: {model_name} with params: {model_params}")
    return model
