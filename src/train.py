"""Main script for GRPO training on GSM8K dataset."""

import torch

from src.config import (
    get_reasoning_markers,
    get_system_prompt,
    get_training_config,
)
from src.data import (
    create_prompt_format,
    create_regex_patterns,
    load_gsm8k_dataset,
)
from src.inference import format_chat_prompt, generate_response
from src.model import add_lora_adapters, initialize_model, save_model
from src.rewards import (
    check_answer,
    check_numbers,
    match_format_approximately,
    match_format_exactly,
)


def train(
    model_name: str = "unsloth/gemma-3-1b-it",
    max_seq_length: int = 1024,
    max_prompt_length: int = 256,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    full_finetuning: bool = False,
    output_dir: str = "outputs",
    save_model_path: str = "gemma-3",
    log_completions: bool = True,
    wandb_project: str = "gsm8k-grpo",
) -> None:
    """
    Train a model using GRPO on the GSM8K dataset.

    Args:
        model_name: Name of the model to fine-tune
        max_seq_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        load_in_4bit: Whether to load in 4bit quantization
        load_in_8bit: Whether to load in 8bit quantization
        full_finetuning: Whether to do full finetuning
        output_dir: Directory to save training outputs
        save_model_path: Path to save the final model
        log_completions: Whether to log completions to wandb
        wandb_project: Name of the wandb project
    """

    # Initialize wandb
    import wandb
    wandb.init(project=wandb_project)
    
    from trl import GRPOConfig, GRPOTrainer
    # Initialize model and tokenizer
    model, tokenizer = initialize_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        full_finetuning=full_finetuning,
    )

    # Add LoRA adapters
    model = add_lora_adapters(model)

    # Get markers and format system prompt
    markers = get_reasoning_markers()
    system_prompt = get_system_prompt(markers)

    # Load and process dataset
    dataset = load_gsm8k_dataset(split="train")
    formatted_dataset = create_prompt_format(system_prompt, markers, dataset)

    # Create regex patterns for reward functions
    patterns = create_regex_patterns(markers)

    # Set up training arguments
    training_config = get_training_config(
        max_prompt_length=max_prompt_length,
        max_seq_length=max_seq_length,
    )
    training_config["output_dir"] = output_dir
    training_config["log_completions"] = log_completions
    training_config["report_to"] = ["wandb"]
    training_args = GRPOConfig(**training_config)

    # Initialize reward functions with the correct patterns and markers
    reward_functions = [
        lambda completions, **kwargs: match_format_exactly(
            completions, pattern=patterns["format"], **kwargs
        ),
        lambda completions, **kwargs: match_format_approximately(
            completions, markers=markers, **kwargs
        ),
        lambda prompts, completions, answer, **kwargs: check_answer(
            prompts, completions, answer, pattern=patterns["format"], **kwargs
        ),
        lambda prompts, completions, answer, **kwargs: check_numbers(
            prompts, completions, answer, pattern=patterns["numbers"], **kwargs
        ),
    ]

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=formatted_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    save_model(model, tokenizer, save_model_path)

    # Finish wandb run
    wandb.finish()

    print(f"Model successfully trained and saved to {save_model_path}")


def inference_demo(
    model_path: str = "gemma-3",
    query: str = "What is the sqrt of 101?",
    max_new_tokens: int = 64,
) -> None:
    """
    Run a simple inference demo with the trained model.

    Args:
        model_path: Path to the saved model
        query: Question to ask the model
        max_new_tokens: Maximum number of tokens to generate
    """
    # Initialize model and tokenizer
    model, tokenizer = initialize_model(
        model_name=model_path,
        max_seq_length=1024,
    )

    # Get markers and format system prompt
    markers = get_reasoning_markers()
    system_prompt = get_system_prompt(markers)

    # Format prompt and generate
    formatted_prompt = format_chat_prompt(tokenizer, system_prompt, query)
    generate_response(
        model,
        tokenizer,
        formatted_prompt,
        max_new_tokens=max_new_tokens,
    )


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model
    train()

    # Run inference demo
    inference_demo()
