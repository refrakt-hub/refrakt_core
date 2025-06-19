from typing import Any, Dict
from omegaconf import OmegaConf

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

        # Convert DictConfig to regular dict if needed
        if hasattr(model_params, "items"):
            model_params_dict = dict(model_params)
        else:
            model_params_dict = model_params

        if wrapper_name:
            print(f"[INFO] Attempting to use wrapper '{wrapper_name}' for model '{model_name}'")

            # Step 1: Instantiate the raw model
            raw_model = get_model_fn(model_name, **model_params_dict).to(device)

            # Step 2: Extract extra args for wrapper (e.g., variant)
            wrapper_args = {k: v for k, v in model_params_dict.items() if k not in ("in_channels", "num_classes", "input_dim", "hidden_dim")}

            # Step 3: Instantiate the wrapper
            model = load_wrapper(wrapper_name, model=raw_model, **wrapper_args).to(device)

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
