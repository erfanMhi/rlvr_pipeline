import logging
import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.profiler as profiler
from datasets import Dataset

# TrainingArguments for type checks or if GRPOConfig is replaced.
# Also for accessing enums/constants for complex TrainingArgs interactions.
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)

from src.components.training_loop.interface import TrainingLoopInterface

logger = logging.getLogger(__name__)


# --- Helper for Profiler (from src/train.py) ---
def get_trace_handler(
    phase: str, base_trace_dir: str = "./profiler_logs"
) -> Any:
    """Creates a TensorBoard trace handler for a phase-specific subdir."""
    logdir = os.path.join(base_trace_dir, phase)
    os.makedirs(logdir, exist_ok=True)
    logger.info(f"Profiler trace handler for '{phase}' writing to: {logdir}")
    return profiler.tensorboard_trace_handler(logdir)


# --- DefaultTrainingLoopComponent Implementation ---
class DefaultTrainingLoopComponent(TrainingLoopInterface):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # self.config holds params for GRPOConfig and profiler, e.g.:
        # "output_dir", "learning_rate", "max_prompt_length",
        # "grpo_num_generations", "profile_enabled", etc.

    def validate_config(self) -> bool:
        required_args = [
            "output_dir",
            "learning_rate",
            "max_steps",
            "per_device_train_batch_size",
            "max_prompt_length",
            "max_completion_length",
        ]
        for arg in required_args:
            if arg not in self.config:
                logger.error(f"Missing required training arg in config: {arg}")
                return False

        max_prompt_length = self.config.get("max_prompt_length")
        max_completion_length = self.config.get("max_completion_length")

        if not isinstance(max_prompt_length, int) or not isinstance(
            max_completion_length, int
        ):
            logger.error(
                "'max_prompt_length' and 'max_seq_length' must be integers."
            )
            return False

        return True

    def train(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        reward_functions: Optional[List[Callable]],
        callbacks: Optional[List[TrainerCallback]],
    ) -> None:
        if not self.validate_config():
            raise ValueError("Training loop configuration is invalid.")

        from trl import (  # Specific trainer and config for GRPO
            GRPOConfig,
            GRPOTrainer,
        )

        # --- Prepare GRPOConfig ---
        # Params are from self.config, matching src/train.py structure.
        grpo_config_args = {
            "output_dir": self.config["output_dir"],
            "report_to": self.config.get(
                "report_to",
                ["wandb"] if self.config.get("log_to_wandb") else [],
            ),
            "logging_steps": self.config.get("logging_steps", 10),
            "learning_rate": self.config["learning_rate"],
            "optim": self.config.get("optim_name", "adamw_torch_fused"),
            "adam_beta1": self.config.get("adam_beta1", 0.9),
            "adam_beta2": self.config.get("adam_beta2", 0.999),
            "weight_decay": self.config.get("weight_decay", 0.01),
            "lr_scheduler_type": self.config.get(
                "lr_scheduler_type", "cosine"
            ),
            "warmup_ratio": self.config.get("warmup_ratio", 0.03),
            "max_grad_norm": self.config.get("max_grad_norm", 1.0),
            "max_steps": self.config["max_steps"],
            "per_device_train_batch_size": self.config[
                "per_device_train_batch_size"
            ],
            "gradient_accumulation_steps": self.config.get(
                "gradient_accumulation_steps", 1
            ),
            "save_steps": self.config.get("save_steps", 500),
            # Evaluation is handled by specific callbacks
            "max_prompt_length": self.config["max_prompt_length"],
            "num_generations": self.config.get("grpo_num_generations", 10),
            "log_completions": self.config.get(
                "log_completions_to_wandb", False
            ),
            "max_completion_length": self.config["max_completion_length"],
            "remove_unused_columns": self.config.get(
                "remove_unused_columns", False
            ),
            "dataloader_num_workers": self.config.get(
                "dataloader_num_workers", 0
            ),
            "seed": self.config.get("seed", 42),
            # "fp16": self.config.get("fp16", torch.cuda.is_available()),
        }

        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and self.config.get("use_bf16", False)
        ):
            grpo_config_args["bf16"] = True
            grpo_config_args["fp16"] = False  # bf16 and fp16 are exclusive
            logger.info("BF16 training enabled.")
        # elif grpo_config_args["fp16"]:
        #     logger.info("FP16 training enabled.")

        try:
            training_args = GRPOConfig(**grpo_config_args)
        except TypeError as e:
            logger.error(f"Error initializing GRPOConfig: {e}")
            logger.error(f"GRPOConfig arguments: {grpo_config_args}")
            raise

        if not reward_functions:
            logger.warning(
                "No reward functions for GRPOTrainer. GRPOTrainer may fail."
            )
            reward_functions = []  # GRPOTrainer expects a list

        logger.info("Initializing GRPOTrainer...")
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_functions,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=callbacks if callbacks else [],
        )

        profile_enabled = self.config.get("profile_enabled", False)
        profiler_trace_dir = self.config.get(
            "profiler_trace_dir", "./profiler_logs"
        )

        if profile_enabled:
            logger.info(
                f"Profiling enabled. Traces to TensorBoard: "
                f"{profiler_trace_dir}/train."
            )
            activities = [
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ]
            schedule_config = self.config.get(
                "profiler_schedule",
                {"wait": 1, "warmup": 1, "active": 3, "repeat": 1},
            )
            schedule = profiler.schedule(**schedule_config)

            with profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=get_trace_handler(
                    "train", base_trace_dir=profiler_trace_dir
                ),
                record_shapes=self.config.get("profiler_record_shapes", True),
                profile_memory=self.config.get(
                    "profiler_profile_memory", False
                ),
                with_stack=self.config.get("profiler_with_stack", False),
            ) as _:  # Assign to _ if not used
                logger.info("Starting GRPOTrainer.train() with profiler...")
                trainer.train()
                # To export trace manually (optional):
                # trace_path = os.path.join(
                #     profiler_trace_dir, "train_trace.json"
                # )
                # _.export_chrome_trace(trace_path)
        else:
            logger.info("Starting GRPOTrainer.train()...")
            trainer.train()

        logger.info("Training loop completed.")
