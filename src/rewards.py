"""Reward functions for GRPO training on GSM8K dataset."""

import re
from typing import Any, Dict, List


def match_format_exactly(
    completions: List[List[Dict[str, str]]], pattern: re.Pattern, **kwargs: Any
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
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if pattern.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(
    completions: List[List[Dict[str, str]]],
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
        response = completion[0]["content"]
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
    debug: bool = False,
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
    if debug:
        print(
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
