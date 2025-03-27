"""Data loading and processing for GSM8K dataset."""

import re
from typing import Any, Callable, Dict, List, Optional, Union

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


def load_math_dataset(split: str = "test") -> Any:
    """
    Load the MATH dataset (competition math).

    Args:
        split: Dataset split to load (only 'test' and 'train' available)

    Returns:
        The dataset object
    """
    # Note: The MATH dataset is large.
    # Consider streaming or subsampling if needed.
    # MATH is unavailable due to DMCA takedown.
    # return load_dataset("hendrycks/math", split=split)
    raise NotImplementedError(
        "The hendrycks/math dataset is currently unavailable on Hugging Face "
        "Hub. Consider using SVAMP or another dataset."
    )


def load_svamp_dataset(split: str = "test") -> Any:
    """
    Load the SVAMP dataset.

    Args:
        split: Dataset split to load (only 'test' and 'train' available).

    Returns:
        The dataset object.
    """
    # Check Hugging Face Hub for the exact name if this doesn't work,
    # common names are ChilleD/SVAMP or svamp
    try:
        return load_dataset("ChilleD/SVAMP", split=split)
    except Exception as e:
        print(f"Could not load ChilleD/SVAMP: {e}. Trying 'svamp'...")
        try:
            # Fallback in case the canonical name changes or is just 'svamp'
            return load_dataset("svamp", split=split)
        except Exception as e2:
            raise ValueError(
                "Could not load SVAMP dataset using 'ChilleD/SVAMP' "
                "or 'svamp' names."
            ) from e2


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


def extract_svamp_answer(answer_value: Union[int, float]) -> Optional[str]:
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


# --- Helper functions for create_prompt_format ---


def _get_gsm8k_question(example: Dict[str, str]) -> str:
    """Extracts the question from a GSM8K example."""
    return example["question"]


def _get_gsm8k_answer(answer_text: Any) -> Optional[str]:
    """Extracts the answer from GSM8K answer text."""
    # GSM8K answer extractor expects string input
    return extract_hash_answer(str(answer_text))


def _get_math_question(example: Dict[str, str]) -> str:
    """Extracts the problem text from a MATH example."""
    return example["problem"]


def _get_math_answer(solution_text: Any) -> Optional[str]:
    """Extracts the answer from MATH solution text."""
    # MATH answer extractor expects string input
    return extract_boxed_answer(str(solution_text))


def _get_svamp_question(example: Dict[str, Any]) -> str:
    """Combines Body and Question for a SVAMP example."""
    return f"{example['Body']}\nQuestion: {example['Question']}"


# ----------------------------------------------


def create_prompt_format(
    system_prompt: str, reasoning_markers: Dict[str, str], dataset: Any
) -> Any:
    """Format the dataset with prompts and extracted answers.

    Handles GSM8K ('question', 'answer'), MATH ('problem', 'solution'),
    and SVAMP ('Body' + 'Question', 'Answer') formats.

    Args:
        system_prompt: System prompt to use.
        reasoning_markers: Dict of markers for reasoning/solution sections
        dataset: The dataset to format

    Returns:
        The formatted dataset
    """
    # Determine dataset type based on columns
    cols = dataset.column_names
    question_col_data: Callable[[Dict[str, Any]], str]
    answer_extractor: Callable[[Any], Optional[str]]
    answer_col: str

    if "question" in cols and "answer" in cols and "Body" not in cols:  # GSM8K
        question_col_data = _get_gsm8k_question
        answer_col = "answer"
        answer_extractor = _get_gsm8k_answer
    elif (
        "problem" in cols and "solution" in cols
    ):  # MATH (currently unavailable)
        question_col_data = _get_math_question
        answer_col = "solution"
        answer_extractor = _get_math_answer
    elif "Body" in cols and "Question" in cols and "Answer" in cols:  # SVAMP
        question_col_data = _get_svamp_question
        answer_col = "Answer"
        answer_extractor = extract_svamp_answer
    else:
        raise ValueError(
            "Dataset columns do not match expected GSM8K, MATH, or SVAMP "
            f"formats. Found columns: {cols}"
        )

    def format_example(
        example: Dict[str, Any],
    ) -> Dict[str, Union[List, str, None]]:
        """Helper to format a single example."""
        # Handle potential None values in the answer column if any
        raw_answer_data = example.get(answer_col)
        extracted_answer = (
            answer_extractor(raw_answer_data)
            if raw_answer_data is not None
            else None
        )
        # Get the question/problem text using the determined function
        question_text = question_col_data(example)

        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ],
            "answer": extracted_answer,
        }

    # Remove examples where the answer could not be extracted
    # for training/evaluation
    original_count = len(dataset)
    formatted_dataset = dataset.map(format_example)
    # Filter out null answers AFTER mapping to ensure consistency
    filtered_dataset = formatted_dataset.filter(
        lambda x: x["answer"] is not None
    )
    filtered_count = len(filtered_dataset)
    if filtered_count < original_count:
        print(
            f"Filtered out {original_count - filtered_count} examples"
            " with unextractable/invalid answers."
        )

    return filtered_dataset


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
