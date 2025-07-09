"""
Improved logging configuration for Refrakt with better safety and no global variables.

This module provides a thread-safe, context-aware logging system that avoids
global variables and provides better error handling and configuration management.
"""

import logging
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class RefraktLoggingManager:
    """
    Thread-safe logging manager that avoids global variables and provides
    context-aware logging with better configuration management.
    """

    _instance: Optional["RefraktLoggingManager"] = None
    _lock = threading.Lock()
    _initialized: bool  # Add this line for type annotation

    def __new__(cls) -> "RefraktLoggingManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._loggers: Dict[str, logging.Logger] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._default_config = {
            "log_dir": "./logs",
            "console": True,
            "debug": False,
            "log_types": [],
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5,
        }
        self._initialized = True

    def configure_logger(
        self,
        name: str,
        log_dir: Optional[str] = None,
        console: Optional[bool] = None,
        debug: Optional[bool] = None,
        log_types: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> logging.Logger:
        """
        Configure a logger with the specified settings.

        Args:
            name: Logger name
            log_dir: Directory for log files
            console: Whether to log to console
            debug: Enable debug logging
            log_types: Types of logging to enable
            **kwargs: Additional configuration options

        Returns:
            Configured logger instance
        """
        # Merge with default config
        config = self._default_config.copy()
        if log_dir is not None:
            config["log_dir"] = log_dir
        if console is not None:
            config["console"] = console
        if debug is not None:
            config["debug"] = debug
        if log_types is not None:
            config["log_types"] = log_types
        config.update(kwargs)

        # Check if logger exists and if we need to reconfigure it
        existing_logger = self._loggers.get(name)
        existing_config = self._configs.get(name, {})

        # If logger exists and config hasn't changed, return existing logger
        if existing_logger and existing_config == config:
            return existing_logger

        # If logger exists but config has changed, remove it to reconfigure
        if existing_logger:
            self.remove_logger(name)

        # Create logger
        logger = logging.getLogger(f"refrakt:{name}")
        logger.setLevel(logging.DEBUG if config["debug"] else logging.INFO)
        logger.propagate = False

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        simple_formatter = logging.Formatter("%(message)s")

        # File handler
        if config["log_dir"]:
            log_path = Path(str(config["log_dir"])) / name
            log_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = log_path / f"{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

        # Console handler
        if config["console"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(simple_formatter)
            logger.addHandler(console_handler)

        # Store configuration
        self._loggers[name] = logger
        self._configs[name] = config

        return logger

    def get_logger(self, name: str) -> logging.Logger:
        """Get a configured logger by name."""
        if name not in self._loggers:
            return self.configure_logger(name)
        return self._loggers[name]

    def get_config(self, name: str) -> Dict[str, Any]:
        """Get the configuration for a logger."""
        return self._configs.get(name, self._default_config.copy())

    def update_config(self, name: str, **kwargs: Any) -> None:
        """Update configuration for a logger."""
        if name in self._configs:
            self._configs[name].update(kwargs)
        else:
            self._configs[name] = self._default_config.copy()
            self._configs[name].update(kwargs)

    def remove_logger(self, name: str) -> None:
        """Remove a logger and its configuration."""
        if name in self._loggers:
            logger = self._loggers[name]
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            del self._loggers[name]
        if name in self._configs:
            del self._configs[name]

    def list_loggers(self) -> List[str]:
        """List all configured loggers."""
        return list(self._loggers.keys())

    @contextmanager
    def temporary_logger(self, name: str, **config: Any) -> Any:
        """Context manager for temporary logger configuration."""
        original_logger = self._loggers.get(name)
        original_config = self._configs.get(name)

        try:
            if name in self._loggers:
                self.remove_logger(name)
            yield self.configure_logger(name, **config)
        finally:
            if name in self._loggers:
                self.remove_logger(name)
            if original_logger:
                self._loggers[name] = original_logger
            if original_config:
                self._configs[name] = original_config


# Global logging manager instance
_logging_manager = RefraktLoggingManager()


def get_logging_manager() -> RefraktLoggingManager:
    """Get the global logging manager instance."""
    return _logging_manager


def configure_logger(
    name: str,
    log_dir: Optional[str] = None,
    console: Optional[bool] = None,
    debug: Optional[bool] = None,
    log_types: Optional[List[str]] = None,
    **kwargs: Any,
) -> logging.Logger:
    """Configure a logger with the specified settings."""
    return _logging_manager.configure_logger(
        name, log_dir, console, debug, log_types, **kwargs
    )


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger by name."""
    return _logging_manager.get_logger(name)


def get_config(name: str) -> Dict[str, Any]:
    """Get the configuration for a logger."""
    return _logging_manager.get_config(name)


def update_config(name: str, **kwargs: Any) -> None:
    """Update configuration for a logger."""
    _logging_manager.update_config(name, **kwargs)


def remove_logger(name: str) -> None:
    """Remove a logger and its configuration."""
    _logging_manager.remove_logger(name)


def list_loggers() -> List[str]:
    """List all configured loggers."""
    return _logging_manager.list_loggers()


@contextmanager
def temporary_logger(name: str, **config: Any) -> Any:
    """Context manager for temporary logger configuration."""
    with _logging_manager.temporary_logger(name, **config) as logger:
        yield logger


# Convenience function for backward compatibility
def get_global_logger() -> logging.Logger:
    """Get the default global logger for backward compatibility."""
    return get_logger("default")


def set_global_logger(logger: logging.Logger) -> None:
    """Set a custom global logger for backward compatibility."""
    _logging_manager._loggers["default"] = logger


def reset_global_logger() -> None:
    """Reset the global logger for backward compatibility."""
    _logging_manager.remove_logger("default")
