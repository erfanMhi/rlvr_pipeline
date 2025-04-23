from abc import abstractmethod
from typing import Any, Dict, Optional

from src.components.base_component import BaseComponent

# Potentially define common event types or context objects here
# class PipelineEvent:
#     def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
#         self.name = name
#         self.context = context or {}


class ObserverInterface(BaseComponent):
    """Interface for observer components that react to pipeline events."""

    @abstractmethod
    def on_pipeline_start(self, orchestrator_config: Dict[str, Any]) -> None:
        """Called when the main pipeline/orchestrator starts."""
        pass

    @abstractmethod
    def on_pipeline_end(
        self, status: str, error: Optional[Exception] = None
    ) -> None:
        """Called when the main pipeline/orchestrator ends."""
        pass

    @abstractmethod
    def on_step_start(
        self, step_name: str, step_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called when a specific pipeline stage starts."""
        pass

    @abstractmethod
    def on_step_end(
        self,
        step_name: str,
        output: Optional[Any] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Called when a specific pipeline stage ends."""
        pass

    # Example for more specific events:
    # @abstractmethod
    # def log_metrics(self, metrics, step=None):
    #     """Log specific metrics."""
    #     pass

    # @abstractmethod
    # def log_completions(self, prompts, completions, step=None):
    #     """Log model completions."""
    #     pass
