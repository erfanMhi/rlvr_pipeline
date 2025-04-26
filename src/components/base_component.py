from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseComponent(ABC):
    """Base class for all components in the training pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Potentially initialize a logger here if all components need one
        # import logging
        # self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the component-specific configuration."""
        pass
