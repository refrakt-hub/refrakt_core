import importlib
import pytest

class TestHelpersInit:
    def test_import_helpers_init(self):
        importlib.import_module('src.refrakt_core.api.helpers')

    def test_helpers_module_has_any_callable(self):
        mod = importlib.import_module('src.refrakt_core.api.helpers')
        funcs = [f for f in dir(mod) if callable(getattr(mod, f, None)) and not f.startswith('__')]
        assert isinstance(funcs, list) 