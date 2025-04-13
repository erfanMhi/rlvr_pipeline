"""Configuration parameters for GRPO training."""

from typing import Any, Dict

# New comprehensive model configuration structure.
# For more complex configurations, consider using ``TypedDict`` for improved
# type safety.
ModelConfig = Dict[
    str, Dict[str, str]
]  # Type for an individual model entry, e.g., {"markers": {...}}
TaskConfig = Dict[
    str, str
]  # Type for an individual task entry, e.g., {"system_prompt_template": "..."}

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "unsloth/gemma-3-1b-it": {
        "markers": {
            "reasoning_start": "<start_working_out>",
            "reasoning_end": "<end_working_out>",
            "solution_start": "<SOLUTION>",
            "solution_end": "</SOLUTION>",
        },
    }
}


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "financial_reasoning": {
        "system_prompt_template": (
            "You are given a financial problem. "
            "First, provide your reasoning and thought process to arrive at "
            "the final answer. Place your reasoning between {reasoning_start} "
            "and {reasoning_end}.\n"
            "Then, provide the final answer to the problem based on the "
            "provided information directly between {solution_start} and "
            "{solution_end}."
        ),
    },
    "math_reasoning": {
        "system_prompt_template": (
            "You are given a problem.\n"
            "Think about the problem and provide your working out.\n"
            "Place it between {reasoning_start} and {reasoning_end}.\n"
            "Then, provide your solution between {solution_start} "
            "and {solution_end}"
        ),
    },
}


def get_task_config(task_name: str) -> TaskConfig:
    """
    Retrieve the full configuration for a given task_name.

    Args:
        task_name: The identifier for the task configuration.

    Returns:
        The configuration dictionary for the specified task.

    Raises:
        ValueError: If the task_name is not found.
    """
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Configuration for task_name '{task_name}' not found."
        )
    return TASK_CONFIGS[task_name]


def get_model_config(model_id: str) -> ModelConfig:
    """
    Retrieve the full configuration for a given model_id.

    Args:
        model_id: The identifier for the model configuration.

    Returns:
        The configuration dictionary for the specified model.

    Raises:
        ValueError: If the model_id is not found.
    """
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Configuration for model_id '{model_id}' not found.")
    return MODEL_CONFIGS[model_id]


def get_markers_for_model(model_id: str) -> Dict[str, str]:
    """
    Get markers for a specific model.

    Args:
        model_id: The identifier for the model.

    Returns:
        Dictionary of marker strings for the specified model.
    """
    config = get_model_config(model_id)
    markers = config.get("markers")
    if not isinstance(markers, dict):
        # This case should ideally not happen if configs are well-formed
        raise TypeError(f"Markers for model '{model_id}' are not a dict.")
    # Further check if all values in markers dict are strings if necessary
    # Type assertion for linters/type checkers after isinstance check
    return markers  # type: ignore[return-value]


def generate_system_prompt_for_model(task_name: str, **kwargs: Any) -> str:
    """Generate a system prompt for a specific model.

    Uses its template and markers.

    Args:
        model_id: The identifier for the model.
        **kwargs: Additional context to format into the prompt template.

    Returns:
        Formatted system prompt string.
    """
    config = get_task_config(task_name)
    template_any = config.get("system_prompt_template")
    markers_any = config.get("markers")

    if not isinstance(template_any, str):
        raise TypeError(
            f"System prompt template for task '{task_name}' is not a string."
        )
    if not isinstance(markers_any, dict):
        raise TypeError(f"Markers for task '{task_name}' are not a dict.")

    # After type checks, we can assert their types for the formatter
    template: str = template_any
    markers: Dict[str, str] = markers_any  # type: ignore[assignment]

    combined_context = {**markers, **kwargs}
    return template.format(**combined_context)


# Updated/Legacy functions
def get_reasoning_markers() -> Dict[str, str]:
    """
    Get standard markers for reasoning and solution sections.

    This function uses the 'default_v1' model configuration.

    Returns:
        Dictionary of marker strings
    """
    return get_markers_for_model("default_v1")


def get_system_prompt(markers: Dict[str, str] | None = None) -> str:
    """
    Generate system prompt (uses 'default_v1' model configuration).

    The 'markers' argument is kept for compatibility but ideally not used
    if relying on the model_id to fetch the correct markers.

    Args:
        markers: Optional dictionary of marker strings. If None, uses default.

    Returns:
        Formatted system prompt
    """
    if markers:
        # This branch handles direct marker passing for legacy compatibility
        # but might not align with a specific model's template if markers
        # are arbitrary.
        # A more robust solution would be to ensure `markers` match a known
        # template structure or disallow arbitrary marker sets here if using
        # model-specific templates.
        default_template_any = MODEL_CONFIGS["default_v1"].get(
            "system_prompt_template"
        )
        if not isinstance(default_template_any, str):
            raise TypeError("Default template is not a string.")
        return default_template_any.format(**markers)
    return generate_system_prompt_for_model("default_v1")
