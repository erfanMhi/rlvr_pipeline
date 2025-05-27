from abc import abstractmethod
from typing import Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from src.components.base_component import BaseComponent


class DataComponentInterface(BaseComponent):
    """Interface for data handling components."""

    @abstractmethod
    def load_and_prepare_data(
        self,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str,
        split: str = "train",
        dataset_name_override: Optional[str] = None,
    ) -> Dataset:
        """Loads, processes, and formats the dataset.

        Returns:
            A HuggingFace Dataset object.
        """
        pass
