import importlib
import os

from refrakt_core.registry.model_registry import register_model

__all__ = ["register_model"]


def auto_import_models() -> None:
    model_dir = os.path.dirname(__file__)
    for filename in os.listdir(model_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = f"{__name__}.{filename[:-3]}"
            importlib.import_module(module_name)


auto_import_models()
