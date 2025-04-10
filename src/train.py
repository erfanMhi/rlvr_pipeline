"""Main script for GRPO training on GSM8K dataset."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.profiler as profiler
import wandb
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.config import generate_system_prompt_for_model, get_markers_for_model
from src.data import create_regex_patterns, get_dataset_processor
from src.evaluation import evaluate_model
from src.inference import format_chat_prompt, generate_response
from src.model import add_lora_adapters, initialize_model, save_model
from src.rewards import get_reward_pipelines

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_trace_handler(phase: str) -> Any:
    """Creates a TensorBoard trace handler for a phase-specific subdir."""
    base_trace_dir = "./profiler_logs"
    logdir = os.path.join(base_trace_dir, phase)
    os.makedirs(logdir, exist_ok=True)
    logger.info(
        f"Profiler trace handler configured for phase '{phase}' "
        f"writing to: {logdir}"
    )
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

        logger.info(f"\n--- Evaluating at step {state.global_step} ---")
        model.eval()

        for dataset_name in self.eval_datasets:
            eval_trace_handler = None
            if self.profile:
                phase = f"eval_{dataset_name}_step_{state.global_step}"
                eval_trace_handler = get_trace_handler(phase)

            # Get the appropriate dataset processor
            try:
                processor = get_dataset_processor(dataset_name)
                # Prepare the evaluation dataset
                # Assuming 'test' split for evaluation
                eval_dataset = processor.create_formatted_dataset(
                    system_prompt=self.system_prompt, split="test"
                )
                if (
                    self.eval_num_samples is not None
                    and self.eval_num_samples > 0
                ):
                    eval_dataset = eval_dataset.select(
                        range(min(self.eval_num_samples, len(eval_dataset)))
                    )

            except ValueError as e:
                error_msg = (
                    f"Could not get processor for {dataset_name}: {e}. "
                    f"Skipping."
                )
                logger.warning(error_msg)
                continue
            except NotImplementedError as e:
                logger.warning(
                    f"Dataset {dataset_name} not implemented: {e}. Skipping."
                )
                continue

            if not eval_dataset or len(eval_dataset) == 0:
                logger.info(
                    f"No data for {dataset_name} at step "
                    f"{state.global_step}. Skipping."
                )
                continue

            accuracy = evaluate_model(
                model=model,
                tokenizer=self.tokenizer,
                dataset_name_for_logging=dataset_name,
                eval_dataset=eval_dataset,
                markers=self.markers,
                max_new_tokens=self.eval_max_new_tokens,
                eval_batch_size=self.eval_batch_size,
                trace_handler=eval_trace_handler,
            )

            # Log the accuracy to wandb
            wandb.log(
                {f"eval/{dataset_name}_accuracy": accuracy},
                step=state.global_step,
            )

        logger.info(
            f"--- Evaluation finished for step {state.global_step} ---"
        )
        model.train()
        self._last_log_step = state.global_step

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        model = kwargs.get("model")
        if model is not None:
            self._run_evaluation(args, state, model)
        else:
            logger.warning(
                "Warning: Model not found in kwargs for initial evaluation."
            )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            model = kwargs.get("model")
            if model is not None:
                self._run_evaluation(args, state, model)
            else:
                logger.warning(
                    f"Warning: Model not found in kwargs for eval at step "
                    f"{state.global_step}."
                )


# Helper to determine problem configuration based on dataset name
def _get_problem_config(training_dataset_name: str) -> Tuple[str, str]:
    """Determines model_config_id and problem_name from training_dataset_name."""
    dataset_name_lower = training_dataset_name.lower()
    if "finqa" in dataset_name_lower:
        return "finqa_v1", "finqa"
    if "gsm8k" in dataset_name_lower:
        return "default_v1", "gsm8k"

    logger.warning(
        f"Could not determine problem config from '{training_dataset_name}'. "
        f"Using default 'default_v1' and problem 'default'."
    )
    return "default_v1", "default"  # Fallback


def train(
    # Model parameters
    model_name: str,
    max_seq_length: int,
    max_prompt_length: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    # Data parameters
    training_dataset_name: str,
    eval_datasets: List[str],
    # Optimizer parameters
    optim_name: str,
    learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    weight_decay: float,
    # Training parameters
    full_finetuning: bool,
    trainer_output_dir: str,
    save_model_path: str,
    profile: bool,
    warmup_ratio: float,
    lr_scheduler_type: str,
    logging_steps: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    save_steps: int,
    max_grad_norm: float,
    report_to: List[str],
    log_completions_to_wandb: bool,
    # GRPO parameters
    grpo_num_generations: int,
    # Evaluation parameters
    evaluation_enabled: bool,
    eval_steps: int,
    eval_num_samples: Optional[int],
    eval_batch_size: int,
    eval_max_new_tokens: int,
    # Wandb parameters
    wandb_project: str,
) -> None:
    """Train a model using GRPO.

    Args:
        model_name: Name of the model to fine-tune.
        max_seq_length: Max sequence length for model and tokenizer.
        max_prompt_length: Max prompt length for GRPO.
        load_in_4bit: Whether to load in 4bit quantization.
        load_in_8bit: Whether to load in 8bit quantization.
        training_dataset_name: Name of the dataset for training.
        eval_datasets: List of dataset names for evaluation.
        optim_name: Optimizer name (e.g., 'adamw_torch_fused').
        learning_rate: Learning rate for the optimizer.
        adam_beta1: AdamW beta1.
        adam_beta2: AdamW beta2.
        weight_decay: Weight decay for the optimizer.
        full_finetuning: Whether to do full finetuning.
        trainer_output_dir: Dir for trainer CPs (relative to hydra CWD).
        save_model_path: Path to save final model (relative to hydra CWD).
        profile: Whether to enable PyTorch Profiler.
        warmup_ratio: Warmup ratio for learning rate scheduler.
        lr_scheduler_type: Learning rate scheduler type.
        logging_steps: Log every N steps.
        per_device_train_batch_size: Batch size per device for training.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_steps: Total number of training steps.
        save_steps: Save checkpoint every N steps.
        max_grad_norm: Maximum gradient norm for clipping.
        report_to: List of services to report results to (e.g., ["wandb"]).
        log_completions_to_wandb: For GRPO, log completions to wandb.
        grpo_num_generations: Number of generations for GRPO.
        evaluation_enabled: Whether to enable evaluation callback.
        eval_steps: Evaluate every N steps.
        eval_num_samples: Number of samples for evaluation (None for all).
        eval_batch_size: Batch size for evaluation inference.
        eval_max_new_tokens: Max new tokens for eval generation.
        wandb_project: Name of the wandb project.
    """
    wandb.init(project=wandb_project)

    from trl import GRPOConfig, GRPOTrainer

    model_config_id, problem_name_for_rewards = _get_problem_config(
        training_dataset_name
    )

    with torch.cuda.nvtx.range("Initialize Model and Tokenizer"):
        model, tokenizer = initialize_model(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            full_finetuning=full_finetuning,
        )

    with torch.cuda.nvtx.range("Add LoRA Adapters"):
        model = add_lora_adapters(model)

    with torch.cuda.nvtx.range("Prepare Data and Prompts"):
        markers = get_markers_for_model(model_config_id)
        system_prompt = generate_system_prompt_for_model(model_config_id)

        try:
            processor = get_dataset_processor(training_dataset_name)
            # CRITICAL: Ensure the FinQA dataset processor prepares the
            # 'answer' column as: List[Dict[str, str]], where each dict is
            # {'gold_program': str, 'gold_answer': str}. This is vital for
            # the finqa_reward_adapted function in rewards.py.
            formatted_dataset = processor.create_formatted_dataset(
                system_prompt, split="train"
            )
        except (ValueError, NotImplementedError) as e:
            error_msg = (
                f"Error preparing training dataset "
                f"{training_dataset_name}:\n{e}"
            )
            logger.error(error_msg)
            wandb.finish()
            return

        if not formatted_dataset or len(formatted_dataset) == 0:
            logger.warning(
                f"No training data for {training_dataset_name}. Aborting."
            )
            wandb.finish()
            return

        # For FinQA, OP_RE is defined in rewards.py and used internally by its
        # rewards.
        # GSM8K rewards use patterns["format"] and patterns["numbers"].
        patterns = create_regex_patterns(markers)

    training_args_dict = {
        "output_dir": trainer_output_dir,
        "learning_rate": learning_rate,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "optim": optim_name,
        "logging_steps": logging_steps,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "max_grad_norm": max_grad_norm,
        "report_to": report_to,
        "max_prompt_length": max_prompt_length,
        "num_generations": grpo_num_generations,
        "log_completions": log_completions_to_wandb,
        "max_completion_length": max_seq_length - max_prompt_length,
    }
    training_args = GRPOConfig(**training_args_dict)

    # Get reward functions using the factory from rewards.py
    try:
        reward_functions = get_reward_pipelines(
            problem_name=problem_name_for_rewards,
            patterns=patterns,
            markers=markers,
        )
    except ValueError as e:
        logger.error(f"Error getting reward functions: {e}. Aborting.")
        wandb.finish()
        return

    if not reward_functions:
        logger.error(
            f"No reward functions for problem '{problem_name_for_rewards}'. "
            "Aborting."
        )
        wandb.finish()
        return

    with torch.cuda.nvtx.range("Setup Evaluation Callback"):
        callbacks = []
        if evaluation_enabled and eval_datasets:
            # Ensure system_prompt and markers for eval are also from
            # model_config_id
            eval_system_prompt = generate_system_prompt_for_model(
                model_config_id
            )
            eval_markers = get_markers_for_model(model_config_id)
            evaluation_callback = EvaluationCallback(
                eval_datasets=eval_datasets,
                eval_steps=eval_steps,
                system_prompt=eval_system_prompt,
                markers=eval_markers,
                tokenizer=tokenizer,
                eval_max_new_tokens=eval_max_new_tokens,
                eval_num_samples=eval_num_samples,
                eval_batch_size=eval_batch_size,
                profile=profile,
            )
            callbacks.append(evaluation_callback)

    with torch.cuda.nvtx.range("Initialize Trainer"):
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_functions,
            args=training_args,
            train_dataset=formatted_dataset,
            callbacks=callbacks,
        )

    with torch.cuda.nvtx.range("Training"):
        if profile:
            logger.info(
                "Profiling enabled. Traces will be logged to TensorBoard."
            )
            with profiler.profile(
                activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA,
                ],
                schedule=profiler.schedule(
                    wait=1, warmup=1, active=1, repeat=1
                ),
                on_trace_ready=get_trace_handler("train"),
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            ):
                trainer.train()
        else:
            trainer.train()

    with torch.cuda.nvtx.range("Save Model"):
        save_model(model, tokenizer, save_model_path)

    wandb.finish()
    logger.info(f"Model successfully trained and saved to {save_model_path}")


def inference_demo(
    model_path: str = "gemma-3",
    query: str = "What is the sqrt of 101?",
    max_new_tokens: int = 64,
    model_id: str = "default_v1",
) -> None:
    """Demonstrates inference with a trained model.

    Args:
        model_path: Path to the saved model.
        query: The query string to ask the model.
        max_new_tokens: Maximum number of new tokens to generate.
        model_id: Identifier for the model configuration to use.
    """
    logger.info(f"Starting inference demo with model: {model_path}")
    logger.info(f"Query: {query}")

    model, tokenizer = initialize_model(
        model_name=model_path,
        max_seq_length=1024,  # Placeholder, adjust as needed
        load_in_4bit=False,
        load_in_8bit=False,
        # Not relevant for loading a pre-trained model for inference
        full_finetuning=False,
    )
    model.eval()  # Set model to evaluation mode

    # Get system prompt for the given model_id.
    # Markers are not directly used by format_chat_prompt
    system_prompt = generate_system_prompt_for_model(model_id)
    # We also need the markers if we want to extract solution later,
    # even if not passed to generation
    markers = get_markers_for_model(model_id)

    formatted_prompt = format_chat_prompt(
        tokenizer=tokenizer,  # Added tokenizer
        system_prompt=system_prompt,  # Renamed from system_message
        user_prompt=query,  # Renamed from user_message, removed markers
    )
    logger.info("Formatted prompt for model:")
    logger.info(formatted_prompt)

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        formatted_prompt=formatted_prompt,  # Renamed from prompt
        max_new_tokens=max_new_tokens,
        # Removed markers argument
    )
    logger.info("Model response:")
    logger.info(response)

    # Solution extraction logic remains, using markers obtained earlier
    solution_start = markers.get("solution_start")
    solution_end = markers.get("solution_end")

    if (
        response
        and solution_start
        and solution_end
        and solution_start in response
    ):
        try:
            start_index = response.rindex(solution_start) + len(solution_start)
            end_index = response.rindex(solution_end, start_index)
            solution_text = response[start_index:end_index].strip()
            logger.info("Extracted solution:")
            logger.info(solution_text)
        except ValueError:
            logger.warning(
                "Could not find solution end marker after start marker. "
                "Full response used."
            )
    else:
        logger.info(
            "Solution markers not found or not configured; "
            "showing full response as solution."
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(
        "This script is intended to be run via train_gsm8k.py using Hydra."
    )
    logger.info(
        "If you want to run train directly, "
        "uncomment and modify an example call."
    )
