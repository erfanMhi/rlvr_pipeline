import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import PreTrainedTokenizerBase

from src.components.data.interface import DataComponentInterface

# We will move AbstractDatasetProcessor and its subclasses here or to
# a utils/data_processors.py. For now, let's assume they are defined in a way
# that can be imported. This might require moving the contents
# of src.data (processors) into this directory.

logger = logging.getLogger(__name__)


# --- Abstract Dataset Processor Definition ---
class AbstractDatasetProcessor(ABC):
    """Defines the abstract interface for dataset-specific processing.

    Subclasses are responsible for fetching raw data, extracting relevant
    fields (like question and answer), and performing any dataset-specific
    parsing or cleaning. It provides a standardized way for the
    `DataComponent` to obtain processable examples without needing to know
    the intricacies of each dataset's structure.
    """

    def __init__(
        self,
        processor_config: Dict[str, Any],
        dataset_name: str,  # This is the actual path/name for load_dataset
        dataset_config_name: Optional[str],
    ):
        self._processor_config = processor_config
        self._dataset_name = dataset_name
        self._dataset_config_name = dataset_config_name

    def validate_config(self) -> bool:
        # Placeholder for validating _processor_config if needed
        return True

    @abstractmethod
    def load_raw_dataset(self, split: str = "train") -> Dataset:
        """Loads the raw dataset from the source."""
        pass

    @abstractmethod
    def get_question_text(self, example: Dict[str, Any]) -> str:
        """Extracts the primary question/prompt text from a raw example."""
        pass

    @abstractmethod
    def extract_ground_truth_answer_from_example(
        self, example: Dict[str, Any]
    ) -> Optional[str]:
        """Extracts and formats the ground truth answer from a raw example."""
        pass

    @staticmethod
    def _extract_boxed_answer(text: str) -> Optional[str]:
        r"""Extract the answer from \boxed{} markers (MATH format)."""
        matches = list(re.finditer(r"\\boxed{(.*?)}", text))
        if matches:
            return matches[-1].group(1).strip()
        return None

    @staticmethod
    def _extract_numeric_answer_from_field(
        answer_value: Any,
    ) -> Optional[str]:
        """Extracts a string representation of a numeric answer.

        Handles integers and floats, converting float to int if it's whole.
        """
        if answer_value is None:
            return None
        if isinstance(answer_value, float) and answer_value.is_integer():
            return str(int(answer_value))
        return str(answer_value)

    def format_single_example_for_model(
        self, system_prompt: str, example: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Formats a single raw example into the model's expected input.

        structure.

        Args:
            system_prompt: The system prompt to prepend.
            example: A single example from the raw dataset.

        Returns:
            A dictionary with "prompt" (list of role/content dicts) and
            "answer" (string), or None if the example is invalid.
        """
        question_text = self.get_question_text(example)
        ground_truth_answer = self.extract_ground_truth_answer_from_example(
            example
        )

        if ground_truth_answer is None:
            logger.debug(
                f"Could not extract answer for example: "
                f"{example.get('id', 'N/A')} in {self.__class__.__name__}"
            )
            return None

        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ],
            "answer": ground_truth_answer,
        }

    def create_formatted_dataset_internal(
        self, system_prompt: str, split: str = "train"
    ) -> Dataset:
        """Loads, processes, and formats the dataset for a given split."""
        raw_dataset = self.load_raw_dataset(split=split)
        original_count = len(raw_dataset)
        processed_examples = []
        for example in raw_dataset:
            formatted_example = self.format_single_example_for_model(
                system_prompt, example
            )
            if formatted_example:
                processed_examples.append(formatted_example)

        if not processed_examples:
            logger.warning(
                f"No processable examples found for {self.__class__.__name__} "
                f"on split {split}. Returning empty dataset."
            )
            # Define features for an empty dataset to avoid errors downstream
            empty_features = Features(
                {
                    "prompt": Sequence(
                        feature={
                            "role": Value("string"),
                            "content": Value("string"),
                        }
                    ),
                    "answer": Value("string"),
                }
            )
            return Dataset.from_list([], features=empty_features)

        formatted_dataset = Dataset.from_list(processed_examples)
        filtered_count = len(formatted_dataset)
        if filtered_count < original_count:
            logger.info(
                f"Filtered out {original_count - filtered_count} examples "
                f"from {self.__class__.__name__} on split {split} due to "
                "unextractable/invalid answers."
            )
        return formatted_dataset


# --- Concrete Dataset Processors ---
class GSM8KDatasetProcessor(AbstractDatasetProcessor):
    """Implements dataset-specific processing for the GSM8K dataset.

    It handles loading the raw GSM8K data, extracting questions, and
    parsing the specific '#### <answer>' format for ground truth answers.
    """

    @staticmethod
    def _extract_hash_answer(text: str) -> Optional[str]:
        """Extracts answer from '#### <answer>' format."""
        match = re.search(r"####\s*(.+)$", text)
        return match.group(1).strip() if match else None

    def load_raw_dataset(self, split: str = "train") -> Dataset:
        return load_dataset(
            self._dataset_name, self._dataset_config_name, split=split
        )

    def get_question_text(self, example: Dict[str, str]) -> str:
        return example["question"]

    def extract_ground_truth_answer_from_example(
        self, example: Dict[str, Any]
    ) -> Optional[str]:
        return self._extract_hash_answer(str(example.get("answer", "")))


class FinQADatasetProcessor(AbstractDatasetProcessor):
    """Implements dataset-specific processing for the FinQA dataset.

    Handles loading FinQA data, constructing the question from its
    structured components (pre_text, post_text, table), and extracting
    the answer.
    """

    def load_raw_dataset(self, split: str = "train") -> Dataset:
        if split == "validation":
            logger.warning(
                "FinQA does not have a validation split. Using 'test' "
                "split instead for processor %s.",
                self.__class__.__name__,
            )
            split = "test"
        return load_dataset(
            self._dataset_name, self._dataset_config_name, split=split
        )

    def get_question_text(self, example: Dict[str, Any]) -> str:
        pre_text = "\n".join(example.get("pre_text", []))
        post_text = "\n".join(example.get("post_text", []))
        table_rows = example.get("table", [])
        table_str = "\n".join(["\t".join(row) for row in table_rows])
        question = example.get("qa", {}).get("question", "")
        return f"{pre_text}\n{table_str}\n{post_text}\nQuestion: {question}"

    def extract_ground_truth_answer_from_example(
        self, example: Dict[str, Any]
    ) -> Optional[str]:
        answer = example.get("qa", {}).get("answer")
        if answer is not None:
            return str(answer).strip()
        return None


# ... (Other processors like SVAMP, MATH would go here) ...
# For brevity, I'm omitting SVAMP and MATH, but they'd follow the same pattern.

PROCESSOR_REGISTRY: Dict[str, Type[AbstractDatasetProcessor]] = {
    "gsm8k": GSM8KDatasetProcessor,
    "finqa": FinQADatasetProcessor,
    # "svamp": SVAMPDatasetProcessor,
    # "math": MATHDatasetProcessor,
}


class DefaultDataComponent(DataComponentInterface):
    """Orchestrates the data loading and preparation process.

    It selects and utilizes a specific `AbstractDatasetProcessor` based on
    configuration to handle dataset-specifics. Its main responsibilities:
    1. Configuring and managing the lifecycle of a dataset processor.
    2. Providing an interface (`load_and_prepare_data`) for the system to
       get formatted data, delegating the core processing to the chosen
       dataset processor.
    3. Determining the task type (e.g., 'math_reasoning') based on the
       dataset.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._dataset_name_key = self.config.get("dataset_name")
        self._dataset_load_path = self.config.get(
            "dataset_path", self._dataset_name_key
        )
        self._dataset_load_config_name = self.config.get("dataset_config_name")

        if not self._dataset_name_key:
            logger.error(
                "'dataset_name' not specified in DataComponent config."
            )
            self._processor = None
            return

        processor_key = self._dataset_name_key.lower()
        processor_class = PROCESSOR_REGISTRY.get(processor_key)

        if not processor_class:
            logger.error(
                f"No processor found for key '{processor_key}'. Check "
                f"dataset_name ('{self._dataset_name_key}') in config and "
                f"PROCESSOR_REGISTRY."
            )
            self._processor = None
        else:
            processor_specific_config = self.config.get("processor_config", {})
            try:
                self._processor = processor_class(
                    processor_config=processor_specific_config,
                    dataset_name=self._dataset_load_path,
                    dataset_config_name=self._dataset_load_config_name,
                )
            except Exception as e:
                logger.error(
                    f"Failed to instantiate processor "
                    f"{processor_class.__name__} for key '{processor_key}': "
                    f"{e}"
                )
                self._processor = None

    def validate_config(self) -> bool:
        if not self._dataset_name_key:
            # Already logged in __init__ if it was missing there
            return False
        if not self._processor:
            # Processor instantiation failure logged in __init__
            return False
        return self._processor.validate_config()

    def load_and_prepare_data(
        self,
        tokenizer: PreTrainedTokenizerBase,  # Kept for interface compliance
        system_prompt: str,
        split: str = "train",
        dataset_name_override: Optional[str] = None,  # Not currently used
    ) -> Dataset:
        """Loads and formats data using the configured dataset processor.

        The `tokenizer` and `dataset_name_override` are not actively used
        by this component in the current implementation but are kept for
        interface compliance and future extensibility.
        """
        if not self._processor:
            logger.error(
                "DataComponent has no valid processor. Cannot load data."
            )
            # Return an empty dataset with standard features
            empty_features = Features(
                {
                    "prompt": Sequence(
                        feature={
                            "role": Value("string"),
                            "content": Value("string"),
                        }
                    ),
                    "answer": Value("string"),
                }
            )
            return Dataset.from_list([], features=empty_features)

        # The processor uses the system_prompt to format each example
        formatted_dataset = self._processor.create_formatted_dataset_internal(
            system_prompt, split
        )

        return formatted_dataset
