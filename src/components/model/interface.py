from abc import abstractmethod
from typing import Dict, Optional, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.components.base_component import BaseComponent


class ModelComponentInterface(BaseComponent):
    """Interface for model handling components."""

    @abstractmethod
    def initialize_model_and_tokenizer(
        self,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Initializes and returns the model and tokenizer."""
        pass

    @abstractmethod
    def add_adapters(self, model: PreTrainedModel) -> PreTrainedModel:
        """Adds PEFT adapters (e.g., LoRA) to the model if configured."""
        pass

    @abstractmethod
    def get_markers(self) -> Dict[str, str]:
        """Returns model-specific markers.

        E.g., for prompt formatting or answer extraction.
        """
        pass

    @abstractmethod
    def save_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        save_path: str,
        adapter_name: Optional[str] = None,
    ) -> None:
        """Saves the model and tokenizer to the specified path."""
        pass
