"""Configuration parameters for GRPO training."""

from typing import Any, Dict, Union

# New comprehensive model configuration structure.
# For more complex configurations, consider using ``TypedDict`` for improved
# type safety.
PromptConfigValue = Union[Dict[str, str], str]
ModelConfig = Dict[str, PromptConfigValue]

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "default_v1": {
        "markers": {
            "reasoning_start": "<start_working_out>",
            "reasoning_end": "<end_working_out>",
            "solution_start": "<SOLUTION>",
            "solution_end": "</SOLUTION>",
        },
        "system_prompt_template": (
            "You are given a problem.\n"
            "Think about the problem and provide your working out.\n"
            "Place it between {reasoning_start} and {reasoning_end}.\n"
            "Then, provide your solution between {solution_start} "
            "and {solution_end}"
        ),
    },
    "experimental_model_x": {
        "markers": {
            "reasoning_start": "[THINKING]",
            "reasoning_end": "[/THINKING]",
            "solution_start": "[SOLUTION_TEXT]",
            "solution_end": "[/SOLUTION_TEXT]",
            "critique_start": "<CRITIQUE>",
            "critique_end": "</CRITIQUE>",
        },
        "system_prompt_template": (
            "Model X - Advanced Reasoning Protocol Engaged.\n"
            "Problem details will be provided.\n"
            "Record your detailed thought process within {reasoning_start} "
            "and {reasoning_end} markers.\n"
            "Present your final solution clearly between {solution_start} "
            "and {solution_end}."
        ),
    },
    # Add more model configurations here as needed
}


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


def generate_system_prompt_for_model(model_id: str, **kwargs: Any) -> str:
    """Generate a system prompt for a specific model.

    Uses its template and markers.

    Args:
        model_id: The identifier for the model.
        **kwargs: Additional context to format into the prompt template.

    Returns:
        Formatted system prompt string.
    """
    config = get_model_config(model_id)
    template_any = config.get("system_prompt_template")
    markers_any = config.get("markers")

    if not isinstance(template_any, str):
        raise TypeError(
            f"System prompt template for model '{model_id}' is not a string."
        )
    if not isinstance(markers_any, dict):
        raise TypeError(f"Markers for model '{model_id}' are not a dict.")

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


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


def get_standard_system_prompt() -> str:
    """Return a generic system prompt without reasoning markers.

    Useful for *standard* (non-reasoning) evaluation where we do **not** want
    the model to expose its chain of thought. The instruction is kept minimal
    so the model outputs only the final answer.
    """
    return (
        "You are presented with a question. Provide a concise and correct "
        "answer "
        "without revealing your chain of thought."
    )
