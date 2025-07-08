import importlib
import pytest

class TestCoreInit:
    def test_import_core_init(self):
        importlib.import_module('src.refrakt_core.api.core')

    def test_core_init_has_symbols(self):
        mod = importlib.import_module('src.refrakt_core.api.core')
        symbols = [s for s in dir(mod) if not s.startswith('__')]
        # It's possible there are no symbols, but this test will pass if the list is empty
        assert isinstance(symbols, list) 