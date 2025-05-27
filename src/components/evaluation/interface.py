from abc import abstractmethod
from typing import Optional

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerState,
)

from src.components.base_component import BaseComponent
from src.components.data.interface import DataComponentInterface
from src.components.model.interface import ModelComponentInterface


class EvaluationComponentInterface(BaseComponent):
    """Interface for evaluation components or callbacks."""

    @abstractmethod
    def get_trainer_callback(
        self,
        data_component_instance: DataComponentInterface,
        model_component_instance: ModelComponentInterface,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Optional[TrainerCallback]:
        """Returns a TrainerCallback for in-training evaluation."""
        pass

    # Alternatively, or additionally, an explicit evaluate method:
    # @abstractmethod
    # def evaluate(
    #     self,
    #     model: PreTrainedModel,
    #     tokenizer: PreTrainedTokenizerBase,
    #     dataset_name: str, # Name of the dataset to evaluate on
    #     system_prompt: str,
    #     markers: Dict[str, str],
    #     global_step: Optional[int] = None
    # ) -> Dict[str, float]:
    #     """Performs evaluation and returns metrics."""
    #     pass

    @abstractmethod
    def on_evaluation_run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        trainer_state: TrainerState,  # Context like global_step
    ) -> None:
        """Called by orchestrator for a post-training evaluation phase."""
        pass
