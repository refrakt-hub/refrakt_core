"""
Wrapper Registry Module
This module is created to manage a registry of wrappers 
that return a <ModelOutput> type for standardized outputs. 
"""
from typing import Any, Callable, Dict, Type

WRAPPER_REGISTRY: Dict[str, Type[Any]] = {}

def register_wrapper(name: str) -> Callable[[Type[Any]], Type[Any]]:
    def decorator(cls: Type[Any]) -> Type[Any]:
        if name in WRAPPER_REGISTRY:
            print(f"Wrapper '{name}' already registered. Overwriting.")
        WRAPPER_REGISTRY[name] = cls
        return cls
    return decorator

def get_wrapper(name: str) -> Type[Any]:
    if name not in WRAPPER_REGISTRY:
        raise ValueError(f"Wrapper '{name}' not found. Available: {list(WRAPPER_REGISTRY.keys())}")
    return WRAPPER_REGISTRY[name]
