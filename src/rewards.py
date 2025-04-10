"""Reward functions for GRPO training on GSM8K dataset."""

import decimal  # Added for FinQA
import logging  # Added for logging
import re
from typing import (  # Ensure Callable is imported
    Any,
    Callable,
    Dict,
    List,
    Union,
)

logger = logging.getLogger(__name__)  # Added for logging


def match_format_exactly(
    completions: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
    pattern: re.Pattern,
    **kwargs: Any,
) -> List[float]:
    """
    Check if completions exactly match the expected format.

    Rewards completions that match the format exactly with 3 points.

    Args:
        completions: List of model completions
        pattern: Regex pattern to match
        **kwargs: Additional arguments

    Returns:
        List of scores for each completion
    """
    scores = []
    for completion in completions:
        score = 0.0
        if isinstance(completion, list):
            response = completion[0]["content"]
        else:
            response = completion["content"]

        logger.debug(f"Response for exact match: {response}")
        logger.debug(
            f"Pattern for exact match: {pattern.pattern}"
        )  # Log pattern string
        search_result = pattern.search(response)
        logger.debug(f"Search result for exact match: {search_result}")
        # Match if format is seen exactly!
        if search_result is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(
    completions: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
    markers: Dict[str, str],
    **kwargs: Any,
) -> List[float]:
    """
    Check if completions approximately match the expected format.

    Rewards or penalizes based on presence of marker tokens.

    Args:
        completions: List of model completions
        markers: Dictionary of format markers
        **kwargs: Additional arguments

    Returns:
        List of scores for each completion
    """
    scores = []
    reasoning_start = markers["reasoning_start"]
    reasoning_end = markers["reasoning_end"]
    solution_start = markers["solution_start"]
    solution_end = markers["solution_end"]

    for completion in completions:
        score = 0.0
        if isinstance(completion, list):
            response = completion[0]["content"]
        else:
            response = completion["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score += 0.5 if response.count(solution_start) == 1 else -0.5
        score += 0.5 if response.count(solution_end) == 1 else -0.5
        scores.append(score)
    return scores


def check_answer(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    pattern: re.Pattern,
    **kwargs: Any,
) -> List[float]:
    """
    Check if the extracted answer matches the expected answer.

    Rewards correct answers and those close in numerical value.

    Args:
        prompts: List of prompts
        completions: List of model completions
        answer: List of correct answers
        pattern: Regex pattern to extract answers
        **kwargs: Additional arguments

    Returns:
        List of scores for each completion
    """
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        (match.group(1) if (match := pattern.search(r)) is not None else None)
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0.0
        if guess is None:
            scores.append(0.0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    score += 0.5
                elif 0.8 <= ratio <= 1.2:
                    score += 0.25
                else:
                    score -= 1.0  # Penalize wrong answers
            except (ValueError, ZeroDivisionError, TypeError):
                score -= 0.5  # Penalize
        scores.append(score)
    return scores


def check_numbers(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    pattern: re.Pattern,
    **kwargs: Any,
) -> List[float]:
    """
    Extract and check numerical answers from the completions.

    Args:
        prompts: List of prompts
        completions: List of model completions
        answer: List of correct answers
        pattern: Regex pattern to extract numbers
        debug: Whether to print debug information
        **kwargs: Additional arguments

    Returns:
        List of scores for each completion
    """
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        (match.group(1) if (match := pattern.search(r)) is not None else None)
        for r in responses
    ]

    scores = []

    logging.debug(
        "*" * 20,
        f"Question:\n{question}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0.0)
            continue
        # Convert to numbers
        try:
            true_answer_float = float(true_answer.strip())
            guess_float = float(guess.strip())
            scores.append(1.5 if guess_float == true_answer_float else 0.0)
        except (ValueError, TypeError):
            scores.append(0)
            continue

    return scores


# FinQA specific constants and helper functions
CURRENCY_RE = re.compile(r"[$€,]")
PERCENT_RE = re.compile(r"%")
PARENS_RE = re.compile(r"^\((.*)\)$")
# OP_RE is used by FinQA's structural_reward and to extract program lines
OP_RE = re.compile(r"^(add|subtract|multiply|divide|average|max|min|change)\(")


def _canon(num: str) -> decimal.Decimal:
    """Canonicalize a number string to Decimal."""
    num_str = str(num).strip()  # Ensure input is string and stripped
    num_str = CURRENCY_RE.sub("", num_str)
    num_str = PERCENT_RE.sub("", num_str)
    if m := PARENS_RE.match(num_str):  # (3.5) -> -3.5
        num_str = "-" + m.group(1)
    try:
        return decimal.Decimal(num_str)
    except decimal.InvalidOperation:
        # Handle cases where conversion might fail after stripping.
        # Returning Decimal(0) if it's an invalid operation after stripping.
        return decimal.Decimal(0)


# Helper function to extract answer from FinQA completion
def _extract_answer_from_finqa_completion(
    response_content: Union[List[Dict[str, str]], str], markers: Dict[str, str]
) -> str:
    """Extracts answer from FinQA model completion string using markers."""
    if isinstance(response_content, list):
        content = response_content[0]["content"]
    else:
        content = response_content
    model_extracted_answer_str = content
    sol_start = markers.get("solution_start")
    sol_end = markers.get("solution_end")

    if sol_start and sol_start in content:
        try:
            start_idx = content.index(sol_start) + len(sol_start)
            try:
                # Attempt to find sol_end *after* sol_start
                end_idx = content.rindex(sol_end, start_idx)
                model_extracted_answer_str = content[start_idx:end_idx]
            except ValueError:  # sol_end not found after sol_start
                model_extracted_answer_str = content[start_idx:]
        except ValueError:  # sol_start not found
            # model_extracted_answer_str remains content (default)
            pass  # Or log this case if needed
    return model_extracted_answer_str.strip()


def finqa_reward_adapted(
    prompts: List[str],
    completions: List[str],
    answers: List[str],  # Gold final answer strings
    markers: Dict[str, str],
    pattern: re.Pattern,
    **kwargs: Any,
) -> List[float]:
    """
    Calculates rewards for FinQA tasks based on final answer comparison.

    Args:
        prompts: List of prompts
        completions: List of model output strings
        answers: List of gold final answer strings
        markers: Dictionary of formatting markers
        pattern: Regex pattern to extract numbers
        **kwargs: Additional arguments

    Returns:
        List of scores for each completion
    """
    scores = []

    for i in range(len(completions)):
        response_content = completions[i]
        current_gold_answer_str = answers[i]

        model_extracted_answer_str = _extract_answer_from_finqa_completion(
            response_content, markers
        )

        # Direct Numerical Comparison Logic
        current_answer_score = 0.0
        try:
            pred_val = _canon(model_extracted_answer_str)
            gold_val = _canon(current_gold_answer_str)

            logger.debug(f"Response: {response_content}")
            logger.debug(
                f"Model extracted answer: {model_extracted_answer_str}"
            )
            logger.debug(f"Gold answer: {current_gold_answer_str}")
            logger.debug(f"Canonized pred val: {pred_val}")
            logger.debug(f"Canonized gold val: {gold_val}")

            if pred_val == gold_val:
                current_answer_score = 4.0
            elif gold_val != decimal.Decimal(0):
                ratio = abs(pred_val / gold_val)
                if decimal.Decimal("0.99") <= ratio <= decimal.Decimal("1.01"):
                    current_answer_score = 2.0
                elif (
                    decimal.Decimal("0.95") <= ratio <= decimal.Decimal("1.05")
                ):
                    current_answer_score = 1.0
                else:
                    current_answer_score = -1.0
            elif gold_val == decimal.Decimal(
                0
            ) and pred_val != decimal.Decimal(0):
                current_answer_score = -1.0

        except (decimal.InvalidOperation, ZeroDivisionError, TypeError):
            current_answer_score = -1.0

        current_total_score = current_answer_score

        scores.append(current_total_score)
    return scores


def check_exact_string_answer_finqa(
    prompts: List[str],
    completions: List[str],
    gold_answers: List[str],
    markers: Dict[str, str],
    reward_for_exact_match: float = 2.0,
    **kwargs: Any,
) -> List[float]:
    """
    Checks if extracted answer exactly matches gold answer string.

    Rewards exact match between stripped extracted answer & stripped gold answer.
    """
    scores = []
    for i in range(len(completions)):
        response_content = completions[i]
        current_gold_answer_str = gold_answers[i]

        # Compare stripped versions to handle whitespace issues
        model_extracted_answer_str = _extract_answer_from_finqa_completion(
            response_content, markers
        )

        if model_extracted_answer_str == current_gold_answer_str.strip():
            scores.append(reward_for_exact_match)
        else:
            scores.append(0.0)
            # Optional: For debugging differences
            # logger.debug(
            #     f"Exact FinQA match failed: Ext '{model_extracted_answer_str}'"
            #     f" vs Gold '{current_gold_answer_str.strip()}'"
            # )
    return scores


# Factory function for reward function pipelines
def get_reward_pipelines(
    problem_name: str,
    patterns: Dict[str, re.Pattern],  # Primarily for GSM8K
    markers: Dict[str, str],  # Problem-specific markers
) -> List[Callable[..., Any]]:
    """
    Returns a list of reward functions based on the problem_name.

    Args:
        problem_name: Identifier for the problem (e.g., "gsm8k", "finqa").
        patterns: Compiled regex patterns, mainly used by GSM8K rewards.
        markers: Dictionary of formatting markers for the specific problem.

    Returns:
        A list of callable reward functions.

    Raises:
        ValueError: If problem_name is not recognized.
    """
    if problem_name.lower().startswith("gsm8k"):
        gsm8k_rewards: List[Callable[..., Any]] = [
            lambda completions, **kwargs_in: match_format_exactly(
                completions, pattern=patterns["format"], **kwargs_in
            ),
            lambda completions, **kwargs_in: match_format_approximately(
                completions, markers=markers, **kwargs_in
            ),
            lambda prompts, completions, answer, **kwargs_in: check_answer(
                prompts,
                completions,
                answer,
                pattern=patterns["format"],
                **kwargs_in,
            ),
            lambda prompts, completions, answer, **kwargs_in: check_numbers(
                prompts,
                completions,
                answer,
                pattern=patterns["numbers"],
                **kwargs_in,
            ),
        ]
        return gsm8k_rewards

    elif problem_name.lower().startswith("finqa"):
        finqa_rewards: List[Callable[..., Any]] = [
            lambda completions, **kwargs_in: match_format_exactly(
                completions, pattern=patterns["format"], **kwargs_in
            ),
            lambda completions, **kwargs_in: match_format_approximately(
                completions, markers=markers, **kwargs_in
            ),
            lambda prompts, completions, answer, **kwargs_in: check_exact_string_answer_finqa(
                prompts,
                completions,
                answer,
                markers,
                reward_for_exact_match=2.0,
                **kwargs_in,
            ),
            lambda prompts, completions, answer, **kwargs_in: finqa_reward_adapted(
                prompts,
                completions,
                answer,
                markers=markers,
                patterns=patterns,
                **kwargs_in,
            ),
        ]
        return finqa_rewards

    else:
        raise ValueError(
            f"Unknown problem_name for reward pipeline: {problem_name}"
        )
