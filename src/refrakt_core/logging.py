"""
Thread-safe singleton pattern for global Refrakt logger access and management.

WARNING: 
This file is not up to Pylint's standards. Safer methods will be implemented 
in the future.
"""

import logging
from threading import Lock
from typing import Optional

# Required for global state tracking
_global_logger_instance = None
_logger_lock: Lock = Lock()

def get_global_logger() -> logging.Logger:
    """
    Get the global logger instance. Returns dummy if not initialized yet.
    """
    global _global_logger_instance
    if _global_logger_instance is None:
        # Return fallback logger silently
        return logging.getLogger("refrakt_null_logger")
    return _global_logger_instance

# def get_global_logger() -> "RefraktLogger":
#     """
#     Get the global logger instance. Creates one if it doesn't exist.

#     Returns:
#         RefraktLogger: The global logger instance.
#     """
#     global _global_logger_instance
#     from refrakt_core.api.core.logger import RefraktLogger

#     if _global_logger_instance is None:
#         with _logger_lock:
#             if _global_logger_instance is None:
#                 existing_logger = logging.getLogger("refrakt")
#                 if existing_logger.hasHandlers():
#                     _global_logger_instance = RefraktLogger.__new__(RefraktLogger)
#                     _global_logger_instance.init_from_existing(
#                         existing_logger=existing_logger,
#                         log_dir="./logs",
#                         log_types=[],
#                         console=True,
#                         debug=True,  # or False depending on your needs
#                     )
#                 else:
#                     _global_logger_instance = RefraktLogger(
#                         log_dir="./logs", log_types=[], console=True
#                     )
#     return _global_logger_instance


def set_global_logger(logger: "RefraktLogger") -> None:
    """
    Set a custom global logger instance.

    Args:
        logger (RefraktLogger): Custom logger instance to register globally.
    """
    global _global_logger_instance
    with _logger_lock:
        if _global_logger_instance is not None:
            _global_logger_instance.close()
        _global_logger_instance = logger


def reset_global_logger() -> None:
    """
    Reset the global logger. Useful for cleanup.
    """
    global _global_logger_instance
    with _logger_lock:
        if _global_logger_instance is not None:
            _global_logger_instance.close()
        _global_logger_instance = None
