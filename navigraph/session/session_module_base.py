# session_module_base.py
import logging

from abc import ABC, abstractmethod
from typing import Any, Optional

from navigraph.utils.logging import get_logger


MODULE_REGISTRY = {}

def register_module(cls):
    """Decorator to automatically register a session module."""
    MODULE_REGISTRY[cls.__name__] = cls
    return cls

def get_module_class(name: str):
    """Get the class of a session module by name."""
    return MODULE_REGISTRY.get(name)

class SessionModule(ABC):
    """
    Abstract base class for all session modules.
    Provides common functionality such as default logging and initialization.
    """
    def __init__(self,
                 logger: Optional[logging.Logger] = get_logger(),
                 **kwargs: Any) -> None:

        self.logger = logger

        # Set additional attributes from keyword arguments.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize(self) -> None:
        """
        Default initialization: logs the class name and its attributes.
        """
        self.logger.info(f"Initializing {self.__class__.__name__} with attributes:")
        for attr, value in self.__dict__.items():
            self.logger.info(f"  {attr}: {value}")

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "SessionModule":
        """
        Factory method to create an instance from configuration.
        Subclasses must implement this method.
        """
        pass
