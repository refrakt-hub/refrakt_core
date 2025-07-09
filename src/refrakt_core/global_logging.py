"""
Thread-safe singleton pattern for global Refrakt logger access and management.

This module provides backward compatibility for the global logging system
while using the new safe logging system underneath.
"""

import logging

from refrakt_core.logging_config import get_logger
from refrakt_core.logging_config import reset_global_logger as _reset_global_logger
from refrakt_core.logging_config import (
    set_global_logger as _set_global_logger,
)  # <-- alias to avoid recursion


def get_global_logger() -> logging.Logger:
    """
    Get the global logger instance. Returns dummy if not initialized yet.
    """
    return get_logger("default")


def set_global_logger(logger: logging.Logger) -> None:
    """
    Set a custom global logger instance.

    Args:
        logger (RefraktLogger): Custom logger instance to register globally.
    """
    _set_global_logger(logger)  # <-- call the imported function


def reset_global_logger() -> None:
    """
    Reset the global logger. Useful for cleanup.
    """
    _reset_global_logger()


# Backward compatibility - maintain the same interface
_global_logger_instance = None
_logger_lock = None


def _get_legacy_global_logger() -> logging.Logger:
    """
    Legacy function for backward compatibility.
    """
    return get_global_logger()


def _set_legacy_global_logger(logger: logging.Logger) -> None:
    """
    Legacy function for backward compatibility.
    """
    set_global_logger(logger)


def _reset_legacy_global_logger() -> None:
    """
    Legacy function for backward compatibility.
    """
    reset_global_logger()
