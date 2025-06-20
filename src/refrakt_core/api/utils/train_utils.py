def get_safe_wrapper(wrapper_name, raw_model, model_params, modules, device):
    import inspect
    wrapper_cls = modules["get_wrapper"](wrapper_name)
    sig = inspect.signature(wrapper_cls.__init__)
    valid_args = set(sig.parameters.keys()) - {"self", "model"}
    wrapper_args = {k: v for k, v in model_params.items() if k in valid_args}
    return wrapper_cls(model=raw_model, **wrapper_args).to(device)
