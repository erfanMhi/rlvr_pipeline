from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional

from src.components.base_component import BaseComponent


class RewardComponentInterface(BaseComponent):
    """Interface for reward calculation components."""

    @abstractmethod
    def get_reward_pipelines(
        self,
        model_info: Optional[Dict[str, Any]],
        reward_functions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Callable[..., Any]]:
        """Constructs and returns a list of reward functions (pipelines).

        Args:
            task_name: The name of the task (e.g., 'math_reasoning').
            patterns: Compiled regex patterns, potentially from ModelComponent.

        Returns:
            A list of callable reward functions.
        """
        pass
