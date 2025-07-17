import decimal  # For FinQA
import json
import logging
import re
from functools import partial, update_wrapper
from typing import Any, Callable, Dict, List, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.components.reward.interface import RewardComponentInterface

logger = logging.getLogger(__name__)

# markers key constants
REASONING_START_MARKER_KEY = "reasoning_start"
REASONING_END_MARKER_KEY = "reasoning_end"
SOLUTION_START_MARKER_KEY = "solution_start"
SOLUTION_END_MARKER_KEY = "solution_end"


def _get_completion_content(
    completion: Any,
) -> str:
    """Extracts content from completion, handles GRPOTrainer list format."""
    if isinstance(completion, list):
        if (
            completion
            and isinstance(completion[0], dict)
            and "content" in completion[0]
        ):
            return str(completion[0]["content"])
        else:
            logger.warning(
                f"Unexpected completion item format (list): {completion}"
            )
            return ""
    elif isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    elif isinstance(completion, str):
        return completion
    else:
        logger.warning(
            f"Unexpected completion format: {type(completion)}, {completion}"
        )
        return ""


def match_format_exactly(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    markers: Dict[str, str],
    reward_value: float = 3.0,
    **kwargs: Any,
) -> List[float]:
    scores = []
    if not markers:
        logger.warning("match_format_exactly: No markers provided. Skipping.")
        return [0.0] * len(completions)

    reasoning_start = markers.get(REASONING_START_MARKER_KEY, "")
    reasoning_end = markers.get(REASONING_END_MARKER_KEY, "")
    solution_start = markers.get(SOLUTION_START_MARKER_KEY, "")
    solution_end = markers.get(SOLUTION_END_MARKER_KEY, "")

    match_format = re.compile(
        rf"^[\s]{{0,}}"
        rf"{re.escape(reasoning_start)}.+?{re.escape(reasoning_end)}.*?"
        rf"{re.escape(solution_start)}(.+?){re.escape(solution_end)}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )

    for (
        completion_str_any
    ) in completions:  # GRPOTrainer might pass list of dicts
        completion_str = _get_completion_content(completion_str_any)
        score = 0.0
        search_result = match_format.search(completion_str)
        if search_result is not None:
            score += reward_value
        scores.append(score)
    return scores


def match_format_approximately(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    markers: Dict[str, str],
    per_marker_reward: float = 0.5,
    per_marker_penalty: float = -0.5,
    **kwargs: Any,
) -> List[float]:
    """This function checks if the completion contains the markers."""
    scores = []
    if not markers:
        raise ValueError("No markers provided. Skipping reward calculation.")

    required_marker_keys = [
        REASONING_START_MARKER_KEY,
        REASONING_END_MARKER_KEY,
        SOLUTION_START_MARKER_KEY,
        SOLUTION_END_MARKER_KEY,
    ]

    for completion_str_any in completions:
        # TODO: investigate whether _get_completion_content is necessary
        completion_str = _get_completion_content(completion_str_any)
        score = 0.0
        for marker_key in required_marker_keys:
            marker_to_check = markers.get(marker_key)

            if not marker_to_check:
                raise ValueError(
                    f"Marker {marker_key} not found in markers. "
                    "Skipping reward calculation."
                )

            if completion_str.count(marker_to_check):
                score += per_marker_reward
            else:
                score += per_marker_penalty

        scores.append(score)

    return scores


def _calculate_numerical_score_gsm8k(
    guess_val: float,
    true_val: float,
    correct_reward: float,
    numerical_close_reward_strong: float,
    numerical_close_reward_weak: float,
    wrong_penalty: float,
) -> float:
    """Helper for numerical score calculation in GSM8K."""
    # Check for exact match first
    if guess_val == true_val:
        return correct_reward

    if true_val == 0:
        return wrong_penalty  # Non-zero guess when answer is 0

    ratio = guess_val / true_val
    if 0.9 <= ratio <= 1.1:
        return numerical_close_reward_strong
    elif 0.8 <= ratio <= 1.2:
        return numerical_close_reward_weak
    else:
        return wrong_penalty


def check_answer_gsm8k(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    pattern: Optional[re.Pattern],  # Changed to Optional[re.Pattern]
    correct_reward: float = 3.0,
    stripped_match_reward: float = 1.5,
    numerical_close_reward_strong: float = 0.5,
    numerical_close_reward_weak: float = 0.25,
    wrong_penalty: float = -1.0,
    # unparsable penalty considered equal to wrong penalty for now
    unparseable_penalty: float = -1.0,
    **kwargs: Any,
) -> List[float]:
    scores = []
    if not pattern:
        logger.warning("check_answer_gsm8k: No pattern provided. Skipping.")
        return [unparseable_penalty] * len(completions)

    for completion_str_any, true_answer_str in zip(completions, answer):
        completion_str = _get_completion_content(completion_str_any)
        score = 0.0
        match = pattern.search(completion_str)
        extracted_guess_str = (
            match.group(1) if match and match.group(1) else None
        )
        if extracted_guess_str is None:
            scores.append(unparseable_penalty)
            continue

        true_answer_str = true_answer_str
        if extracted_guess_str == true_answer_str:
            score += correct_reward
        elif extracted_guess_str.strip() == true_answer_str.strip():
            score += stripped_match_reward
        else:
            try:
                guess_val = float(extracted_guess_str.replace(",", ""))
                true_val = float(true_answer_str.replace(",", ""))
                score += _calculate_numerical_score_gsm8k(
                    guess_val,
                    true_val,
                    correct_reward,
                    numerical_close_reward_strong,
                    numerical_close_reward_weak,
                    wrong_penalty,
                )
            except (ValueError, TypeError):
                score += wrong_penalty
        scores.append(score)
    return scores


CURRENCY_RE = re.compile(r"[$€,£¥]")
PERCENT_RE = re.compile(r"%")
PARENS_RE = re.compile(r"^\((.+)\)$")


def _canon_finqa(num_str_in: Any) -> decimal.Decimal:
    num_str = str(num_str_in).strip()

    # Remove currency symbols (including at the end)
    num_str = CURRENCY_RE.sub("", num_str)
    # Remove percent signs
    num_str = PERCENT_RE.sub("", num_str)

    # Handle parentheses for negative numbers (accounting format)
    if m := PARENS_RE.match(num_str):
        num_str = "-" + m.group(1)

    # Remove any remaining whitespace and commas for thousands
    num_str = num_str.replace(" ", "").replace(",", "")

    try:
        return decimal.Decimal(num_str)
    except decimal.InvalidOperation:
        logger.debug(
            f"FinQA _canon: Could not convert '{num_str_in}' to Decimal. "
            "Returning 0."
        )
        return decimal.Decimal(0)


def _extract_answer_from_finqa_completion(
    response_content: str,
    extract_solution_pattern: Optional[re.Pattern],
) -> Optional[str]:
    if not extract_solution_pattern:
        logger.error(
            "Solution extraction pattern missing in "
            "_extract_answer_from_finqa_completion"
        )
        return None

    match = extract_solution_pattern.search(response_content)
    if match and match.group(1):
        return match.group(1).strip()

    logger.debug(
        "FinQA extract: Solution pattern did not match in completion."
    )
    return None


def finqa_numerical_match_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    solution_pattern: Optional[re.Pattern],
    exact_match_reward: float = 3.0,
    approx_match_reward: float = 1.0,
    no_answer_penalty: float = -1.0,
    relative_tolerance: decimal.Decimal = decimal.Decimal("1e-2"),
    **kwargs: Any,
) -> List[float]:
    scores = []
    if not solution_pattern:
        logger.warning(
            "finqa_numerical_match_reward: No solution_pattern provided. "
            "Skipping."
        )
        return [no_answer_penalty] * len(completions)

    for completion_str_any, gold_answer_str in zip(completions, answer):
        completion_str = _get_completion_content(completion_str_any)
        model_answer_str = _extract_answer_from_finqa_completion(
            completion_str, solution_pattern
        )

        if model_answer_str is None:
            scores.append(no_answer_penalty)
            continue

        try:
            gold_val = _canon_finqa(gold_answer_str)
            model_val = _canon_finqa(model_answer_str)

            if model_val == gold_val:
                scores.append(exact_match_reward)
            elif gold_val == decimal.Decimal(0):
                scores.append(no_answer_penalty)
            elif abs(gold_val - model_val) <= relative_tolerance * abs(
                gold_val
            ):
                scores.append(approx_match_reward)
            else:
                scores.append(no_answer_penalty)
        except Exception as e:
            logger.error(
                f"Error in finqa_numerical_match_reward processing: {e}",
                exc_info=True,
            )
            scores.append(no_answer_penalty)
    return scores


def check_numbers(
    prompts: List[Any],  # Using List[Any] due to specific access pattern
    completions: List[Any],
    answer: List[str],
    number_extraction_pattern: re.Pattern,
    no_match_score: float = 0.0,
    match_score: float = 1.5,
    conversion_error_score: float = 0.0,
    **kwargs: Any,
) -> List[float]:
    """
    Checks for a number in the completion after a solution start marker.

    Compares it to the reference answer.
    """
    scores = []
    question_content = ""

    # Attempt to extract question content for debugging output
    # This assumes a specific structure for prompts: List[List[Dict[str, str]]]
    if prompts and isinstance(prompts, list) and len(prompts) > 0:
        first_prompt_item = prompts[0]
        if isinstance(first_prompt_item, list) and len(first_prompt_item) > 0:
            last_message = first_prompt_item[-1]
            if isinstance(last_message, dict):
                question_content = last_message.get("content", "")

    for i, completion_item in enumerate(completions):
        response_str = _get_completion_content(completion_item)

        # Ensure there's a corresponding reference answer
        if i >= len(answer):
            logger.warning(
                "Mismatch between completions and reference_answers length."
            )
            scores.append(no_match_score)  # Or some other default penalty
            continue
        true_answer_str = answer[i]

        match = number_extraction_pattern.search(response_str)
        extracted_guess_str = (
            match.group(1) if match and match.group(1) else None
        )

        if i == 0:  # Log only for the first item for brevity
            logger.debug(
                "%s Question:\n%s\nAnswer:\n%s\nResponse:\n%s\nExtracted:\n%s",
                "*" * 20,
                question_content,
                true_answer_str,
                response_str,
                extracted_guess_str,
            )

        if extracted_guess_str is None:
            scores.append(no_match_score)
            continue
        try:
            true_val = float(true_answer_str.strip().replace(",", ""))
            guess_val = float(extracted_guess_str.strip().replace(",", ""))
            scores.append(
                match_score if guess_val == true_val else no_match_score
            )
        except (ValueError, TypeError):
            scores.append(conversion_error_score)
            continue
    return scores


class DefaultRewardComponent(RewardComponentInterface):
    # _create_regex_patterns_static removed as patterns are now passed in.

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.reward_configs = self.config.get("reward_functions", {})

    def validate_config(self) -> bool:
        if not isinstance(self.reward_configs, (list, ListConfig)):
            logger.error(
                "'reward_functions' in config must be a list or ListConfig."
            )
            return False

        # Validate individual reward function configs
        for i, reward_config in enumerate(self.reward_configs):
            if not isinstance(reward_config, (dict, DictConfig)):
                logger.error(
                    f"Reward function config at index {i} must be a dict or "
                    f"DictConfig, got {type(reward_config)}."
                )
                return False

            # Check required fields for each reward function
            if not reward_config.get("type"):
                logger.error(
                    f"Reward function config at index {i} missing 'type' "
                    f"field."
                )
                return False

            if not reward_config.get("category"):
                logger.error(
                    f"Reward function config at index {i} missing 'category' "
                    f"field."
                )
                return False

            # Validate params if present
            params = reward_config.get("params", {})
            if params and not isinstance(params, (dict, DictConfig)):
                logger.error(
                    f"Reward function params at index {i} must be a dict or "
                    f"DictConfig, got {type(params)}."
                )
                return False

        return True

    def _build_answer_matching_reward_fn(
        self, fn_config: Dict[str, Any], marker: Dict[str, str]
    ) -> Optional[Callable[..., Any]]:
        fn_type = fn_config.get("type")
        params = fn_config.get("params", {})

        # Fetch all markers from self._markers (passed as 'marker')
        reasoning_start_marker = marker.get(REASONING_START_MARKER_KEY, "")
        reasoning_end_marker = marker.get(REASONING_END_MARKER_KEY, "")
        solution_start_marker = marker.get(SOLUTION_START_MARKER_KEY, "")
        solution_end_marker = marker.get(SOLUTION_END_MARKER_KEY, "")

        if (
            fn_type == "gsm8k_answer_check"
            or fn_type == "finqa_numerical_match"
        ):
            if not all(
                [
                    reasoning_start_marker,
                    reasoning_end_marker,
                    solution_start_marker,
                    solution_end_marker,
                ]
            ):
                error_msg = (
                    f"Reward function '{fn_type}' requires all four markers: "
                    f"'{REASONING_START_MARKER_KEY}', "
                    f"'{REASONING_END_MARKER_KEY}', "
                    f"'{SOLUTION_START_MARKER_KEY}', and "
                    f"'{SOLUTION_END_MARKER_KEY}'. One or more are missing."
                )
                raise ValueError(error_msg)

            # Comprehensive pattern including reasoning and solution
            # The captured group (.+?) for the solution is the 1st group.
            comprehensive_pattern_str = (
                rf"^[\s]{{0,}}"
                rf"{re.escape(reasoning_start_marker)}"
                rf".+?{re.escape(reasoning_end_marker)}.*?"
                rf"{re.escape(solution_start_marker)}(.+?)"
                rf"{re.escape(solution_end_marker)}"
                rf"[\s]{{0,}}$"
            )
            answer_extraction_pattern = re.compile(
                comprehensive_pattern_str, flags=re.MULTILINE | re.DOTALL
            )

            if fn_type == "gsm8k_answer_check":
                reward_fn = partial(
                    check_answer_gsm8k,
                    pattern=answer_extraction_pattern,
                    **params,
                )
                update_wrapper(reward_fn, check_answer_gsm8k)
                return reward_fn
            elif fn_type == "finqa_numerical_match":
                reward_fn = partial(
                    finqa_numerical_match_reward,
                    solution_pattern=answer_extraction_pattern,
                    **params,
                )
                update_wrapper(reward_fn, finqa_numerical_match_reward)
                return reward_fn

        elif fn_type == "check_numbers":
            if not solution_start_marker:  # Only needs solution_start
                raise ValueError(
                    f"Reward function 'check_numbers' requires a "
                    f"'{SOLUTION_START_MARKER_KEY}' in markers, but it's "
                    f"missing."
                )

            number_extraction_regex_str = (
                rf"{re.escape(solution_start_marker)}"
                # Match numbers: -123, 123.45, 1,234, 1,234.56, 1.5e6, etc.
                rf".*?(-?\d+(?:,\d{{3}})*(?:\.\d+)?(?:[eE][+-]?\d+)?)"
            )
            number_extraction_pattern = re.compile(
                number_extraction_regex_str, flags=re.MULTILINE | re.DOTALL
            )
            reward_fn = partial(
                check_numbers,
                number_extraction_pattern=number_extraction_pattern,
                **params,
            )
            update_wrapper(reward_fn, check_numbers)
            return reward_fn

        # Log for unknown or misconfigured answer-matching types
        logger.warning(
            f"Unknown or misconfigured answer_matching fn_type: {fn_type}. "
            "No reward function built."
        )
        return None

    def _build_format_checking_reward_fn(
        self,
        fn_config: Dict[str, Any],
        markers: Dict[str, str],
    ) -> Optional[Callable[..., Any]]:
        fn_type = fn_config.get("type")
        params = fn_config.get("params", {})

        if not markers:
            logger.warning(
                f"Reward fn '{fn_type}' needs markers, "
                "but none were provided."
            )
            return None

        if fn_type == "match_format_exactly":
            reward_fn = partial(
                match_format_exactly,
                markers=markers,
                **params,
            )
            update_wrapper(reward_fn, match_format_exactly)
            return reward_fn
        elif fn_type == "match_format_approximately":
            reward_fn = partial(
                match_format_approximately,
                markers=markers,
                **params,
            )
            update_wrapper(reward_fn, match_format_approximately)
            return reward_fn

        return None

    def get_reward_pipelines(
        self,
        model_info: Optional[Dict[str, Any]],
        reward_functions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Callable[..., Any]]:
        if reward_functions is None:
            reward_functions = self.reward_configs

        if not reward_functions:
            logger.warning("No reward functions configured")
            return []

        if model_info is None:
            raise ValueError("model_info is required")

        markers = model_info.get("markers", {})
        if not markers:
            raise ValueError("markers are required")

        pipelines: List[Callable[..., Any]] = []

        fn_prints = OmegaConf.to_container(reward_functions, resolve=True)

        logger.info(
            "Reward functions used:\n%s",
            json.dumps(fn_prints, indent=2, ensure_ascii=False),
        )

        for fn_config in reward_functions:
            category = fn_config.get(
                "category"
            )  # e.g., "answer_matching", "format_checking"
            reward_fn = None
            if category == "answer_matching":
                reward_fn = self._build_answer_matching_reward_fn(
                    fn_config, markers
                )
            elif category == "format_checking":
                reward_fn = self._build_format_checking_reward_fn(
                    fn_config, markers
                )
            else:
                logger.warning(f"Unknown reward category: {category}")

            if reward_fn:
                pipelines.append(reward_fn)
            else:
                logger.warning(
                    f"Could not build reward function for config: {fn_config}"
                )

        logger.info(f"Built {len(pipelines)} reward functions.")
        return pipelines
