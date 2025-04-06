"""Configuration parameters for GRPO training."""

from typing import Any, Dict


def get_reasoning_markers() -> Dict[str, str]:
    """
    Get standard markers for reasoning and solution sections.

    Returns:
        Dictionary of marker strings
    """
    return {
        "reasoning_start": "<start_working_out>",
        "reasoning_end": "<end_working_out>",
        "solution_start": "<SOLUTION>",
        "solution_end": "</SOLUTION>",
    }


def get_system_prompt(markers: Dict[str, str]) -> str:
    """
    Generate system prompt with reasoning markers.

    Args:
        markers: Dictionary of marker strings

    Returns:
        Formatted system prompt
    """
    return (
        f"You are given a problem.\n"
        f"Think about the problem and provide your working out.\n"
        f"Place it between {markers['reasoning_start']} and "
        f"{markers['reasoning_end']}.\n"
        f"Then, provide your solution between {markers['solution_start']}"
        f"{markers['solution_end']}"
    )


def get_training_config(
    max_prompt_length: int = 256,
    max_seq_length: int = 1024,
) -> Dict[str, Any]:
    """
    Get default GRPO training configuration.

    Args:
        max_prompt_length: Maximum length of prompt
        max_seq_length: Maximum sequence length

    Returns:
        Dictionary with training parameters
    """
    num_generations = 16
    return {
        "learning_rate": 5e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch_fused",
        "logging_steps": 1,
        "per_device_train_batch_size": num_generations,
        "gradient_accumulation_steps": 1,
        "num_generations": num_generations,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_seq_length - max_prompt_length,
        "max_steps": 1,
        "save_steps": 50,
        "max_grad_norm": 0.1,
        "report_to": "all",
        "output_dir": "outputs",
    }
