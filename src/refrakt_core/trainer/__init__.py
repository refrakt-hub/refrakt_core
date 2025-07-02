import importlib
import os

__all__ = []


def auto_import_trainers() -> None:
    trainer_dir = os.path.dirname(__file__)
    for filename in os.listdir(trainer_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = f"{__name__}.{filename[:-3]}"
            importlib.import_module(module_name)


auto_import_trainers()
