"""Data loading and processing for GSM8K dataset."""

import re
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset  # type: ignore


def load_gsm8k_dataset(split: str = "train") -> Any:
    """
    Load the GSM8K dataset.

    Args:
        split: Dataset split to load

    Returns:
        The dataset object
    """
    return load_dataset("openai/gsm8k", "main", split=split)


def extract_hash_answer(text: str) -> Optional[str]:
    """
    Extract the answer from the text after the #### marker.

    Args:
        text: Text containing an answer

    Returns:
        The extracted answer or None if not found
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def create_prompt_format(
    system_prompt: str, reasoning_markers: Dict[str, str], dataset: Any
) -> Any:
    """
    Format the dataset with prompts and extracted answers.

    Args:
        system_prompt: System prompt to use
        reasoning_markers: Dictionary of markers for reasoning/solution
        sections
        dataset: The dataset to format

    Returns:
        The formatted dataset
    """

    def format_example(example: Dict[str, str]) -> Dict[str, Union[List, str]]:
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["question"]},
            ],
            "answer": extract_hash_answer(example["answer"]),  # type: ignore
        }

    return dataset.map(format_example)


def create_regex_patterns(
    reasoning_markers: Dict[str, str],
) -> Dict[str, re.Pattern]:
    """
    Create regex patterns for matching reasoning and solution sections.

    Args:
        reasoning_markers: Dictionary containing marker strings

    Returns:
        Dictionary of compiled regex patterns
    """
    reasoning_start = reasoning_markers["reasoning_start"]
    reasoning_end = reasoning_markers["reasoning_end"]
    solution_start = reasoning_markers["solution_start"]
    solution_end = reasoning_markers["solution_end"]

    format_pattern = re.compile(
        rf"^[\s]{{0,}}"
        rf"{reasoning_start}.+?{reasoning_end}.*?"
        rf"{solution_start}(.+?){solution_end}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )

    numbers_pattern = re.compile(
        rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
    )

    return {"format": format_pattern, "numbers": numbers_pattern}
