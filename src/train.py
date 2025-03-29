"""Main script for GRPO training on GSM8K dataset."""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.profiler as profiler
import wandb
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

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
from src.evaluation import evaluate_model
from src.inference import format_chat_prompt, generate_response
from src.model import add_lora_adapters, initialize_model, save_model
from src.rewards import (
    check_answer,
    check_numbers,
    match_format_approximately,
    match_format_exactly,
)


def get_trace_handler(phase: str) -> Any:
    """Creates a TensorBoard trace handler for a phase-specific subdir."""
    # Define the base directory for all profiler logs
    base_trace_dir = "./profiler_logs"

    # Create the specific subdirectory for this phase
    logdir = os.path.join(base_trace_dir, phase)
    os.makedirs(logdir, exist_ok=True)

    # Split the print statement to avoid line length issue
    print(
        f"Profiler trace handler configured for phase '{phase}' "
        f"writing to: {logdir}"
    )
    # Return the TensorBoard handler
    return profiler.tensorboard_trace_handler(logdir)


class EvaluationCallback(TrainerCallback):
    """Callback to evaluate the model on specified datasets during training."""

    def __init__(
        self,
        eval_datasets: List[str],
        eval_steps: int,
        system_prompt: str,
        markers: Dict[str, str],
        tokenizer: Any,
        eval_max_new_tokens: int,
        eval_num_samples: Optional[int] = None,
        eval_batch_size: int = 8,
        profile: bool = False,
    ):
        self.eval_datasets = eval_datasets
        self.eval_steps = eval_steps
        self.system_prompt = system_prompt
        self.markers = markers
        self.tokenizer = tokenizer
        self.eval_max_new_tokens = eval_max_new_tokens
        self.eval_num_samples = eval_num_samples
        self.eval_batch_size = eval_batch_size
        self._last_log_step = -1
        self.profile = profile

    def _run_evaluation(
        self,
        args: TrainingArguments,
        state: TrainerState,
        model: Any,
    ) -> None:
        if state.global_step == self._last_log_step:
            return

        print(f"\n--- Evaluating at step {state.global_step} ---")
        model.eval()

        for dataset_name in self.eval_datasets:
            eval_trace_handler = None
            if self.profile:
                phase = f"eval_{dataset_name}_step_{state.global_step}"
                eval_trace_handler = get_trace_handler(phase)

            accuracy = evaluate_model(
                model=model,
                tokenizer=self.tokenizer,
                dataset_name=dataset_name,
                system_prompt=self.system_prompt,
                markers=self.markers,
                max_new_tokens=self.eval_max_new_tokens,
                eval_batch_size=self.eval_batch_size,
                num_samples=self.eval_num_samples,
                trace_handler=eval_trace_handler,
            )
            wandb.log(
                {f"eval/{dataset_name}_accuracy": accuracy},
                step=state.global_step,
            )

        print(f"--- Evaluation finished for step {state.global_step} ---")
        model.train()
        self._last_log_step = state.global_step

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Evaluate at the beginning of training (step 0)."""
        model = kwargs.get("model")
        if model is not None:
            self._run_evaluation(args, state, model)
        else:
            print("Warning: Model not found in kwargs for initial evaluation.")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Evaluate every `eval_steps`."""
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            model = kwargs.get("model")
            if model is not None:
                self._run_evaluation(args, state, model)
            else:
                print(
                    f"Warning: Model not found in kwargs for eval at step "
                    f"{state.global_step}."
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
    eval_steps: int = 100,
    eval_datasets: List[str] = [
        "svamp",
        "gsm8k",
    ],
    eval_num_samples: Optional[int] = None,
    eval_batch_size: int = 512,
    eval_max_new_tokens: int = 2048,
    profile: bool = False,
) -> None:
    """Train a model using GRPO on the GSM8K dataset.

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
        eval_steps: Evaluate every N steps.
        eval_datasets: List of dataset names ('svamp', 'gsm8k') to evaluate on.
        eval_num_samples: Number of samples for evaluation (None for all).
        eval_batch_size: Batch size for evaluation inference.
        eval_max_new_tokens: Max new tokens for eval generation.
        profile: Whether to enable PyTorch Profiler.
    """
    # Initialize wandb
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

    # Initialize Evaluation Callback
    evaluation_callback = EvaluationCallback(
        eval_datasets=eval_datasets,
        eval_steps=eval_steps,
        system_prompt=system_prompt,
        markers=markers,
        tokenizer=tokenizer,
        eval_max_new_tokens=eval_max_new_tokens,
        eval_num_samples=eval_num_samples,
        eval_batch_size=eval_batch_size,
        profile=profile,
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        # GRPOTrainer uses tokenizer directly, not processing_class
        tokenizer=tokenizer,  # Changed from processing_class
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=formatted_dataset,
        callbacks=[evaluation_callback],  # Added callback
    )

    # Train the model
    # --- Profiling --- #
    if profile:
        print("Profiling enabled. Traces will be logged to TensorBoard.")
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(
                wait=1, warmup=1, active=1, repeat=1
            ),  # Profile only 1 step
            on_trace_ready=get_trace_handler("train"),  # Use helper function
            record_shapes=True,
            profile_memory=False,  # Disable memory profiling
            with_stack=False,  # Already disabled
        ):  # Remove unused variable assignment
            trainer.train()
            # If you need the profiler object later, assign it back:
            # as prof:
            #    trainer.train()
            #    print(prof.key_averages().table(sort_by="cpu_time_total"))
    else:
        trainer.train()
    # --- End Profiling --- #

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
