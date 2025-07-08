import importlib
import os

__all__ = []


def auto_import_losses() -> None:
    loss_dir = os.path.dirname(__file__)
    for filename in os.listdir(loss_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = f"{__name__}.{filename[:-3]}"
            importlib.import_module(module_name)


auto_import_losses()
