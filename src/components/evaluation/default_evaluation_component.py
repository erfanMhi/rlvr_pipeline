import logging
from typing import Any, Dict, List, Optional

import torch
import wandb  # For logging metrics
from datasets import Dataset
from omegaconf import DictConfig, ListConfig
from tqdm.auto import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import TrainingArguments

from src.components.data.interface import DataComponentInterface
from src.components.evaluation.interface import EvaluationComponentInterface
from src.components.model.interface import ModelComponentInterface

logger = logging.getLogger(__name__)


def extract_solution_marker_answer_eval(
    text: str, markers: Dict[str, str]
) -> Optional[str]:
    import re  # Local import

    solution_start = markers.get("solution_start")
    solution_end = markers.get("solution_end")
    if not solution_start or not solution_end:
        return None
    # Using re.DOTALL for multi-line content within markers
    pattern_str = (
        re.escape(solution_start) + r"(.*?)" + re.escape(solution_end)
    )
    match = None
    # Find last occurrence which is typical for final answers
    for m in re.finditer(pattern_str, text, flags=re.DOTALL | re.MULTILINE):
        match = m
    return match.group(1).strip() if match else None


def compare_answers_eval(
    predicted: Optional[str], expected: Optional[str]
) -> bool:
    if predicted is None or expected is None:
        return False
    # Basic comparison, can be enhanced (e.g. numerical for FinQA)
    return predicted.strip() == expected.strip()


@torch.inference_mode()
def _generate_responses_eval(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    eval_batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    model_generations = []
    # Generate responses in batches using tqdm for progress
    for i in tqdm(
        range(0, len(prompts), eval_batch_size),
        desc="Eval Generation",
        leave=False,
    ):
        batch_prompts = prompts[i : i + eval_batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(
            model.device
        )  # type: ignore

        outputs = model.generate(  # type: ignore
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,  # Greedy for consistency in evaluation
        )
        batch_generations = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        model_generations.extend(batch_generations)
    return model_generations


def _determine_predicted_answer_eval(
    generation: str,
    dataset_name_for_logging: str,  # e.g. "gsm8k", "math", "finqa"
    markers: Optional[Dict[str, str]],
) -> Optional[str]:
    # Logic for answer extraction based on dataset characteristics
    if (
        markers
        and markers.get("solution_start")
        and markers.get("solution_end")
    ):
        # General case for datasets with solution markers (GSM8K, FinQA)
        return extract_solution_marker_answer_eval(generation, markers)
    else:
        # Fallback: use full generation if no specific extraction defined
        return generation.strip()


@torch.inference_mode()
def evaluate_model_core(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_dataset: Dataset,  # Pre-formatted HF dataset
    dataset_name_for_logging: str,
    markers: Optional[Dict[str, str]],
    max_new_tokens: int,
    eval_batch_size: int,
) -> float:
    logger.info(f"Core evaluation for {dataset_name_for_logging}...")
    try:
        # Prompts are expected in chat format List[Dict[str,str]]
        # Convert to flat list of strings for generation
        prompts_for_generation = [
            tokenizer.apply_chat_template(
                ex["prompt"],
                tokenize=False,  # type: ignore
                add_generation_prompt=True,
            )
            for ex in eval_dataset
        ]
        ground_truths = [ex["answer"] for ex in eval_dataset]  # type: ignore
    except KeyError as e:
        logger.error(f"Dataset {dataset_name_for_logging} missing field: {e}")
        return 0.0
    except Exception as e:  # Catch any other error during data prep
        logger.error(
            f"Error processing dataset {dataset_name_for_logging}: {e}"
        )
        return 0.0

    if not prompts_for_generation:
        logger.warning(
            f"No prompts to evaluate for {dataset_name_for_logging}."
        )
        return 0.0

    model_generations = _generate_responses_eval(
        model,
        tokenizer,
        prompts_for_generation,  # type: ignore
        eval_batch_size,
        max_new_tokens,
    )

    correct_count = 0
    num_examples = len(ground_truths)

    for i, (gen, expected_answer) in enumerate(
        zip(model_generations, ground_truths)
    ):
        predicted_answer = _determine_predicted_answer_eval(
            gen, dataset_name_for_logging, markers
        )
        is_correct = compare_answers_eval(predicted_answer, expected_answer)
        if is_correct:
            correct_count += 1

        if i < 5:  # Log first few examples for debugging
            log_msg_parts = [
                f"Eval Example {i+1}/{num_examples} "
                f"({dataset_name_for_logging})",
                f"  Generation (snippet): {gen[:100]}...",
                f"  Predicted Answer: {predicted_answer}",
                f"  Expected Answer: {expected_answer} "
                f"| Correct: {is_correct}",
            ]
            logger.debug("\n".join(log_msg_parts))

    accuracy = (
        (correct_count / num_examples) * 100 if num_examples > 0 else 0.0
    )
    logger.info(
        f"Accuracy on {dataset_name_for_logging}: {accuracy:.2f}% "
        f"({correct_count}/{num_examples})"
    )
    return accuracy


# --- Custom Evaluation Callback ---
class CustomEvaluationCallback(TrainerCallback):
    def __init__(
        self,
        data_component: DataComponentInterface,
        model_component: ModelComponentInterface,
        eval_datasets_names: List[str],
        eval_steps: int,
        eval_max_new_tokens: int,
        system_prompt: str,
        tokenizer: PreTrainedTokenizerBase,
        eval_num_samples: Optional[int] = None,
        eval_batch_size: int = 8,
    ):
        self.data_component = data_component
        self.model_component = model_component
        self.eval_datasets_names = eval_datasets_names
        self.eval_steps = eval_steps
        self.eval_max_new_tokens = eval_max_new_tokens
        self.eval_num_samples = eval_num_samples
        self.eval_batch_size = eval_batch_size
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self._last_log_step = -1  # Initialize last log step

    def _run_evaluation(
        self,
        args: TrainingArguments,
        state: TrainerState,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        # Avoid re-evaluation if already done for this step
        # Allow eval at step 0 only if on_train_begin runs it first
        if state.global_step == self._last_log_step or (
            state.global_step == 0 and self._last_log_step != -1
        ):
            return

        # Update step tracking at the beginning to prevent duplicate runs
        current_step = state.global_step
        self._last_log_step = current_step if current_step > 0 else 0

        logger.info(f"EvaluationCallback: Evaluating at step {current_step}")
        model.eval()  # type: ignore

        # Run evaluation for all datasets
        metrics_to_log = self._evaluate_all_datasets(
            model, tokenizer, current_step
        )

        # Log metrics to WandB
        self._log_metrics_to_wandb(metrics_to_log, current_step)

        logger.info(
            f"EvaluationCallback: Finished eval for step {current_step}"
        )
        model.train()  # type: ignore

    def _evaluate_all_datasets(
        self, model: Any, tokenizer: PreTrainedTokenizerBase, current_step: int
    ) -> Dict[str, float]:
        """Evaluate model on all configured datasets."""
        metrics_to_log = {}

        for dataset_name in self.eval_datasets_names:
            try:
                accuracy = self._evaluate_single_dataset(
                    model, tokenizer, dataset_name, current_step
                )
                if accuracy is not None:
                    metric_name = f"eval/{dataset_name}_accuracy"
                    metrics_to_log[metric_name] = float(accuracy)

            except Exception as e:
                logger.error(
                    f"Error during evaluation of {dataset_name} at step "
                    f"{current_step}: {e}",
                    exc_info=True,
                )
                continue

        return metrics_to_log

    def _evaluate_single_dataset(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str,
        current_step: int,
    ) -> Optional[float]:
        """Evaluate model on a single dataset."""
        # Get task name for the current eval dataset
        system_prompt_for_eval = self.system_prompt
        if system_prompt_for_eval is None:
            logger.error(
                f"System prompt for key '{system_prompt_for_eval}' "
                f"not found for {dataset_name}. Skipping."
            )
            return None

        markers_for_eval = self.model_component.get_markers()

        # Load and prepare dataset using DataComponent
        eval_dataset_formatted = self.data_component.load_and_prepare_data(
            tokenizer=tokenizer,
            system_prompt=system_prompt_for_eval,
            dataset_name_override=dataset_name,
            split="test",  # Default to test split for eval
        )

        if self.eval_num_samples and self.eval_num_samples > 0:
            eval_dataset_formatted = eval_dataset_formatted.select(
                range(
                    min(
                        self.eval_num_samples,
                        len(eval_dataset_formatted),
                    )
                )
            )

        if not eval_dataset_formatted or len(eval_dataset_formatted) == 0:
            raise ValueError(
                f"No data for {dataset_name} (step {current_step})."
            )

        accuracy = evaluate_model_core(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset_formatted,
            dataset_name_for_logging=dataset_name,
            markers=markers_for_eval,
            max_new_tokens=self.eval_max_new_tokens,
            eval_batch_size=self.eval_batch_size,
        )

        # Debug logging
        logger.info(f"Computed accuracy for {dataset_name}: {accuracy}")
        return accuracy

    def _log_metrics_to_wandb(
        self, metrics_to_log: Dict[str, float], current_step: int
    ) -> None:
        """Log metrics to WandB in a single batch."""
        if not metrics_to_log:
            return

        logger.info(f"Current global_step: {current_step}")
        logger.info(f"WandB run active: {wandb.run is not None}")

        if wandb.run:  # Log to wandb if active
            # Get WandB's current step to avoid step conflicts
            wandb_step = wandb.run.step
            logger.info(
                f"Logging to WandB: {list(metrics_to_log.keys())} "
                f"at trainer step {current_step}, wandb step {wandb_step}"
            )
            try:
                # Let WandB auto-increment step to avoid conflicts with
                # built-in trainer callbacks
                if wandb.run is not None:
                    wandb.log(metrics_to_log, commit=True)
                    logger.info(
                        f"Successfully logged {len(metrics_to_log)} "
                        f"metrics to WandB (auto step)"
                    )
                else:
                    logger.error("WandB run became None during logging")
            except Exception as e:
                logger.error(f"Failed to log to WandB: {e}")
                # Fallback: log to console
                self._log_metrics_to_console(metrics_to_log, current_step)
        else:
            logger.warning("WandB run is None - metrics not being logged!")
            # Fallback: log to console
            self._log_metrics_to_console(metrics_to_log, current_step)

    def _log_metrics_to_console(
        self, metrics_to_log: Dict[str, float], current_step: int
    ) -> None:
        """Fallback logging to console when WandB is not available."""
        for metric_name, value in metrics_to_log.items():
            print(f"EVAL METRIC: {metric_name}={value} step={current_step}")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        model = kwargs.get("model")
        # Use stored tokenizer instead of getting from kwargs
        tokenizer = kwargs.get("tokenizer") or self.tokenizer
        if model and tokenizer:
            logger.info("EvalCallback: Initial evaluation on_train_begin.")
            self._run_evaluation(args, state, model, tokenizer)
        else:
            logger.warning(
                "EvalCallback: Model/Tokenizer not found for initial eval."
            )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if (
            state.global_step > 0
            and self.eval_steps > 0
            and state.global_step % self.eval_steps == 0
        ):
            model = kwargs.get("model")
            # Use stored tokenizer instead of getting from kwargs
            tokenizer = kwargs.get("tokenizer") or self.tokenizer
            if model and tokenizer:
                self._run_evaluation(args, state, model, tokenizer)
            else:
                logger.warning(
                    f"EvalCallback: Model/Tokenizer not found at step "
                    f"{state.global_step}."
                )


class DefaultEvaluationComponent(EvaluationComponentInterface):

    def __init__(self, config: Dict[str, Any], prompts_config: Dict[str, Any]):
        super().__init__(config)  # eval_config is now self.config
        self.prompts_config = prompts_config
        self.data_component_instance: Optional[DataComponentInterface] = None
        self.model_component_instance: Optional[ModelComponentInterface] = None

    def set_dependencies(self, data_comp: Any, model_comp: Any) -> None:
        """Injects component instances, called by orchestrator."""
        self.data_component_instance = data_comp
        self.model_component_instance = model_comp

    def _validate_prompt_key_exists(
        self,
        key: Optional[str],
        key_name: str,
        system_prompts: Dict[str, str],
        dataset_name: Optional[str] = None,
    ) -> bool:
        """Helper to validate if a prompt key exists in system_prompts."""
        if key and key not in system_prompts:
            ds_info = f" for dataset '{dataset_name}'" if dataset_name else ""
            logger.error(
                f"Prompt key '{key}' (from '{key_name}')"
                f"{ds_info} not found in global system_prompts."
            )
            return False
        return True

    def _validate_all_datasets_have_prompts(
        self,
        eval_datasets_names: List[str],
        eval_dataset_prompt_keys: Dict[str, str],
        default_eval_prompt_key: Optional[str],
    ) -> bool:
        """Helper to ensure all eval datasets can resolve to a prompt key."""
        for name in eval_datasets_names:
            if (
                not eval_dataset_prompt_keys.get(name)
                and not default_eval_prompt_key
            ):
                logger.error(
                    f"Dataset '{name}' has no specific prompt key and no "
                    "'default_eval_prompt_key' is set."
                )
                return False
        return True

    def validate_config(self) -> bool:
        if not self.config.get("enabled", True):
            return True

        eval_datasets_names = self.config.get("eval_datasets_names")
        if not eval_datasets_names:
            logger.warning(
                "No 'eval_datasets_names' in EvalComponent config. "
                "Eval disabled."
            )
            self.config["enabled"] = False
            return True

        # Validate eval_datasets_names is a list
        if not isinstance(eval_datasets_names, (list, ListConfig)):
            logger.error("'eval_datasets_names' must be a list or ListConfig.")
            return False

        eval_steps = self.config.get("eval_steps")
        if not isinstance(eval_steps, int) or eval_steps <= 0:
            logger.error("'eval_steps' must be a positive int for evaluation.")
            return False

        # Validate eval_dataset_prompt_keys if present
        eval_dataset_prompt_keys = self.config.get(
            "eval_dataset_prompt_keys", {}
        )
        if eval_dataset_prompt_keys and not isinstance(
            eval_dataset_prompt_keys, (dict, DictConfig)
        ):
            logger.error(
                "'eval_dataset_prompt_keys' must be a dict or DictConfig."
            )
            return False

        # Validate eval_max_new_tokens if present
        eval_max_new_tokens = self.config.get("eval_max_new_tokens")
        if eval_max_new_tokens is not None and not isinstance(
            eval_max_new_tokens, int
        ):
            logger.error("'eval_max_new_tokens' must be an integer.")
            return False

        # Validate eval_num_samples if present
        eval_num_samples = self.config.get("eval_num_samples")
        if eval_num_samples is not None and not isinstance(
            eval_num_samples, int
        ):
            logger.error("'eval_num_samples' must be an integer.")
            return False

        # Validate eval_batch_size if present
        eval_batch_size = self.config.get("eval_batch_size")
        if eval_batch_size is not None and not isinstance(
            eval_batch_size, int
        ):
            logger.error("'eval_batch_size' must be an integer.")
            return False

        # TODO: validate prompt later
        return True

    def get_trainer_callback(
        self,
        data_component_instance: DataComponentInterface,
        model_component_instance: ModelComponentInterface,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Optional[TrainerCallback]:
        if not self.config.get("enabled", True) or not self.config.get(
            "eval_datasets_names"
        ):
            logger.info(
                "Evaluation disabled or no datasets. No callback created."
            )
            return None
        if not self.validate_config():  # Re-validate before creating callback
            logger.error(
                "EvalComponent config invalid. Cannot create callback."
            )
            return None

        self.set_dependencies(
            data_component_instance, model_component_instance
        )

        system_prompts = self.prompts_config.get("system_prompts", {})
        prompting_mode = self.prompts_config.get("prompting_mode", {})
        system_prompt_str = system_prompts.get(prompting_mode)
        if system_prompt_str is None:
            logger.error(
                f"System prompt for key '{prompting_mode}' not found in "
                "system_prompts."
            )
            return None

        return CustomEvaluationCallback(
            data_component=data_component_instance,
            model_component=model_component_instance,
            eval_datasets_names=self.config["eval_datasets_names"],
            eval_steps=self.config["eval_steps"],
            eval_max_new_tokens=self.config.get("eval_max_new_tokens", 256),
            eval_num_samples=self.config.get("eval_num_samples"),
            eval_batch_size=self.config.get("eval_batch_size", 8),
            system_prompt=system_prompt_str,
            tokenizer=tokenizer,
        )

    def _process_single_dataset_evaluation(
        self,
        dataset_name: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        trainer_state: TrainerState,
    ) -> None:
        """Process evaluation for a single dataset."""
        logger.info(f"Post-training evaluation on: {dataset_name}")
        try:
            # Get prompt configuration
            system_prompt_for_eval = self._get_system_prompt_for_dataset(
                dataset_name
            )
            if system_prompt_for_eval is None:
                return

            # Prepare dataset
            eval_dataset_formatted = self._prepare_evaluation_dataset(
                dataset_name, tokenizer, system_prompt_for_eval
            )
            if not eval_dataset_formatted:
                return

            # Run evaluation
            markers_for_eval = (
                self.model_component_instance.get_markers()  # type: ignore
            )
            accuracy = evaluate_model_core(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset_formatted,
                dataset_name_for_logging=dataset_name,
                markers=markers_for_eval,
                max_new_tokens=self.config.get("eval_max_new_tokens", 256),
                eval_batch_size=self.config.get("eval_batch_size", 8),
            )
            if wandb.run:
                wandb.log(
                    {f"post_train_eval/{dataset_name}_accuracy": accuracy}
                )
        except Exception as e:
            logger.error(
                f"Error in on_evaluation_run for {dataset_name}: {e}",
                exc_info=True,
            )

    def _get_system_prompt_for_dataset(
        self, dataset_name: str
    ) -> Optional[str]:
        """Get system prompt for a specific dataset."""
        system_prompts_dict = self.prompts_config.get("system_prompts", {})
        eval_dataset_prompt_keys_map = self.config.get(
            "eval_dataset_prompt_keys", {}
        )
        default_key = self.config.get("default_eval_prompt_key")

        prompt_key = eval_dataset_prompt_keys_map.get(
            dataset_name, default_key
        )

        if not prompt_key:
            logger.error(
                f"No prompt key for {dataset_name} and no default in "
                "on_evaluation_run. Skipping."
            )
            return None

        system_prompt_for_eval = system_prompts_dict.get(prompt_key)
        if system_prompt_for_eval is None:
            logger.error(
                f"System prompt for key '{prompt_key}' not found "
                f"for {dataset_name} in on_evaluation_run. Skipping."
            )
            return None

        return str(system_prompt_for_eval)

    def _prepare_evaluation_dataset(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str,
    ) -> Optional[Dataset]:
        """Prepare evaluation dataset for a specific dataset."""
        if self.data_component_instance is None:
            logger.error("Data component instance is None")
            return None

        eval_dataset_formatted = (
            self.data_component_instance.load_and_prepare_data(
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                dataset_name_override=dataset_name,
                split="test",
            )
        )
        num_samples = self.config.get("eval_num_samples")
        if num_samples and num_samples > 0:
            eval_dataset_formatted = eval_dataset_formatted.select(
                range(min(num_samples, len(eval_dataset_formatted)))
            )
        if not eval_dataset_formatted or len(eval_dataset_formatted) == 0:
            logger.info(
                f"No data for post-train eval on {dataset_name}. Skipping."
            )
            return None

        return eval_dataset_formatted

    def on_evaluation_run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        trainer_state: TrainerState,
    ) -> None:
        logger.info("DefaultEvaluationComponent.on_evaluation_run called.")
        if not self.config.get("enabled", True) or not self.config.get(
            "eval_datasets_names"
        ):
            logger.info(
                "Eval not enabled or no datasets for on_evaluation_run."
            )
            return

        if (
            not self.data_component_instance
            or not self.model_component_instance
        ):
            logger.error(
                "Data/Model component instances not set for on_evaluation_run."
            )
            return

        for dataset_name in self.config.get("eval_datasets_names", []):
            self._process_single_dataset_evaluation(
                dataset_name, model, tokenizer, trainer_state
            )

        logger.info("DefaultEvaluationComponent.on_evaluation_run finished.")
