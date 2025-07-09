import importlib

import pytest


class TestUtilsInit:
    def test_import_utils_init(self):
        importlib.import_module("src.refrakt_core.api.utils")

    def test_utils_module_has_any_callable(self):
        mod = importlib.import_module("src.refrakt_core.api.utils")
        funcs = [
            f
            for f in dir(mod)
            if callable(getattr(mod, f, None)) and not f.startswith("__")
        ]
        assert isinstance(funcs, list)
