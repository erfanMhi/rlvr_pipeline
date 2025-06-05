from abc import abstractmethod
from typing import Callable, List, Optional

from datasets import Dataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback

from src.components.base_component import BaseComponent


class TrainingLoopInterface(BaseComponent):
    """Interface for training loop components."""

    @abstractmethod
    def train(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        reward_functions: Optional[List[Callable]],
        callbacks: Optional[List[TrainerCallback]],
    ) -> None:
        """Executes the training loop using configuration from self.config."""
        pass
