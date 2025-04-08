"""Data loading and processing for GSM8K dataset."""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

from datasets import Features, Sequence, Value, load_dataset  # type: ignore


def extract_hash_answer(text: str) -> Optional[str]:
    """
    Extract the answer from the text after the #### marker (GSM8K format).

    Args:
        text: Text containing an answer

    Returns:
        The extracted answer or None if not found
    """
    match = re.search(r"####\s*(.+)$", text)
    return match.group(1).strip() if match else None


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""
    Extract the answer from \boxed{} markers (MATH format).

    Args:
        text: Text containing the boxed answer.

    Returns:
        The extracted answer or None if not found.
    """
    # Find the last occurrence of \boxed{...}
    matches = list(re.finditer(r"\\boxed{(.*?)}", text))
    if matches:
        return matches[-1].group(1).strip()
    return None


def extract_svamp_answer_from_field(
    answer_value: Union[int, float],
) -> Optional[str]:
    """
    Extract/format the answer from the SVAMP dataset's 'Answer' field.

    Args:
        answer_value: The numerical answer value.

    Returns:
        The answer formatted as a string, or None if input is invalid.
    """
    if answer_value is None:
        return None
    # Convert potential float like 5.0 to "5"
    if isinstance(answer_value, float) and answer_value.is_integer():
        return str(int(answer_value))
    return str(answer_value)


def extract_solution_marker_answer(
    text: str, markers: Dict[str, str]
) -> Optional[str]:
    """Extract the answer from between solution markers.

    (e.g., <SOLUTION>...</SOLUTION>).
    This is typically for model generated output.

    Args:
        text: Text possibly containing the answer within markers.
        markers: Dictionary with 'solution_start' and 'solution_end' keys.

    Returns:
        The extracted answer or None if not found.
    """
    solution_start = re.escape(
        markers["solution_start"]
    )  # Escape regex special chars
    solution_end = re.escape(
        markers["solution_end"]
    )  # Escape regex special chars
    # Non-greedy match between the markers
    pattern = rf"{solution_start}(.*?){solution_end}"
    # Find last occurrence, as check_answer reward might implicitly do
    match = None
    for m in re.finditer(pattern, text, flags=re.DOTALL | re.MULTILINE):
        match = m

    return match.group(1).strip() if match else None


class AbstractDatasetProcessor(ABC):
    """Abstract base class for dataset processing."""

    @abstractmethod
    def load_raw_dataset(self, split: str = "train") -> Any:
        """Loads the raw dataset from its source."""
        pass

    @abstractmethod
    def get_question_text(self, example: Dict[str, Any]) -> str:
        """Extracts the question text from a raw dataset example."""
        pass

    @abstractmethod
    def extract_ground_truth_answer_from_example(
        self, example: Dict[str, Any]
    ) -> Optional[str]:
        """Extracts the ground truth answer from a raw dataset example."""
        pass

    def format_single_example_for_model(
        self, system_prompt: str, example: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Formats a single raw example into the required structure for the model.

        This includes the extracted ground truth answer.
        Returns None if the answer cannot be extracted.
        """
        question_text = self.get_question_text(example)
        ground_truth_answer = self.extract_ground_truth_answer_from_example(
            example
        )

        if ground_truth_answer is None:
            return None

        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ],
            "answer": ground_truth_answer,
        }

    def create_formatted_dataset(
        self, system_prompt: str, split: str = "train"
    ) -> Any:
        """
        LTE the dataset for model training/evaluation.

        Filters out examples where the answer could not be extracted.
        """
        raw_dataset = self.load_raw_dataset(split=split)

        original_count = len(raw_dataset)

        # Using a list comprehension and then converting to dataset to handle
        # filtering and avoid issues with map and filter on dataset objects
        # directly with Nones
        processed_examples = []
        for example in raw_dataset:
            formatted_example = self.format_single_example_for_model(
                system_prompt, example
            )
            if formatted_example:
                processed_examples.append(formatted_example)

        # Re-create a Hugging Face Dataset from the list of dictionaries
        # This requires knowing the features, which we can infer if all dicts
        # have same keys
        if not processed_examples:
            # Or handle as an error, or return an empty dataset with
            # defined features
            print(
                f"Warning: No processable examples found for "
                f"{self.__class__.__name__} on split {split}."
            )
            # Attempt to return an empty dataset with expected structure.
            # May need robust handling or explicit schema definition.
            # `Dataset.from_list` handles empty lists or use empty list for
            # caller.
            # HF Trainer needs a HF Dataset object.
            from datasets import Dataset  # Local import

            # Create an empty dataset with a schema. Assumes specific
            # prompt/answer structure.
            # Better: dynamic creation, or ensure one valid example for
            # schema inference.
            # Empty `processed_examples` can cause issues if trainer needs
            # non-empty dataset.
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
            formatted_dataset = Dataset.from_list([], features=empty_features)

        else:
            from datasets import Dataset  # Local import

            formatted_dataset = Dataset.from_list(processed_examples)

        filtered_count = len(formatted_dataset)
        if filtered_count < original_count:
            msg = (
                f"Filtered out {original_count - filtered_count} examples"
                f" from {self.__class__.__name__} on split {split} due to "
                "unextractable/invalid answers."
            )
            print(msg)
        return formatted_dataset


class GSM8KDatasetProcessor(AbstractDatasetProcessor):
    """Processor for the GSM8K dataset."""

    def load_raw_dataset(self, split: str = "train") -> Any:
        return load_dataset("openai/gsm8k", "main", split=split)

    def get_question_text(self, example: Dict[str, Any]) -> str:
        return example["question"]  # type: ignore

    def extract_ground_truth_answer_from_example(
        self, example: Dict[str, Any]
    ) -> Optional[str]:
        return extract_hash_answer(str(example.get("answer", "")))


class SVAMPDatasetProcessor(AbstractDatasetProcessor):
    """Processor for the SVAMP dataset."""

    def load_raw_dataset(self, split: str = "test") -> Any:
        try:
            return load_dataset("ChilleD/SVAMP", split=split)
        except Exception as e:
            print(f"Could not load ChilleD/SVAMP: {e}. Trying 'svamp'...")
            try:
                return load_dataset("svamp", split=split)
            except Exception as e2:
                raise ValueError(
                    "Could not load SVAMP dataset using 'ChilleD/SVAMP' "
                    "or 'svamp' names."
                ) from e2

    def get_question_text(self, example: Dict[str, Any]) -> str:
        return f"{example['Body']}\\nQuestion: {example['Question']}"

    def extract_ground_truth_answer_from_example(
        self, example: Dict[str, Any]
    ) -> Optional[str]:
        answer = example.get("Answer")
        if isinstance(answer, (int, float)):
            return extract_svamp_answer_from_field(answer)
        return None


class MATHDatasetProcessor(AbstractDatasetProcessor):
    """Processor for the MATH dataset (currently unavailable)."""

    def load_raw_dataset(self, split: str = "test") -> Any:
        raise NotImplementedError(
            "The hendrycks/math dataset is currently unavailable on "
            "Hugging Face Hub. Consider using SVAMP or another dataset."
        )

    def get_question_text(self, example: Dict[str, Any]) -> str:
        # Assuming 'problem' is the key for question text in MATH dataset
        if "problem" not in example:
            raise ValueError("MATH dataset example missing 'problem' field.")
        return example["problem"]  # type: ignore

    def extract_ground_truth_answer_from_example(
        self, example: Dict[str, Any]
    ) -> Optional[str]:
        # Assuming 'solution' is the key for answer text in MATH dataset
        if "solution" not in example:
            # Or return None if this is acceptable
            raise ValueError(
                "MATH example missing 'solution' field for answer extraction."
            )
        return extract_boxed_answer(str(example["solution"]))


DATASET_PROCESSORS: Dict[str, Type[AbstractDatasetProcessor]] = {
    "gsm8k": GSM8KDatasetProcessor,
    "svamp": SVAMPDatasetProcessor,
    "math": MATHDatasetProcessor,
}


def get_dataset_processor(dataset_name: str) -> AbstractDatasetProcessor:
    """
    Factory function to get a dataset processor instance.

    Args:
        dataset_name: The name of the dataset.

    Returns:
        An instance of the appropriate dataset processor.

    Raises:
        ValueError: If the dataset_name is not supported.
    """
    processor_class = DATASET_PROCESSORS.get(dataset_name.lower())
    if processor_class:
        return processor_class()
    msg = (
        f"Unsupported dataset: {dataset_name}. "
        f"Available: {list(DATASET_PROCESSORS.keys())}"
    )
    raise ValueError(msg)


def create_regex_patterns(
    reasoning_markers: Dict[str, str],
) -> Dict[str, re.Pattern]:
    """
    Create regex patterns for matching reasoning and solution sections.

    This is generally independent of the input dataset's format and pertains
    to the expected structure of a model's output.

    Args:
        reasoning_markers: Dictionary containing marker strings

    Returns:
        Dictionary of compiled regex patterns
    """
    reasoning_start = re.escape(reasoning_markers["reasoning_start"])
    reasoning_end = re.escape(reasoning_markers["reasoning_end"])
    solution_start = re.escape(reasoning_markers["solution_start"])
    solution_end = re.escape(reasoning_markers["solution_end"])

    format_pattern = re.compile(
        rf"^[\\s]{{0,}}"
        rf"{reasoning_start}.+?{reasoning_end}.*?"
        rf"{solution_start}(.+?){solution_end}"
        rf"[\\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )

    # Pattern to extract numbers from within the model's solution block.
    # Specific to how answers are found in completions.
    numbers_pattern = re.compile(
        rf"{solution_start}[^\d]*([\d\.]+)",
        flags=re.MULTILINE | re.DOTALL,
    )

    return {"format": format_pattern, "numbers": numbers_pattern}
