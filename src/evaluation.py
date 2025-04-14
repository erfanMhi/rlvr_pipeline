"""Model evaluation script."""

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.profiler as profiler
from tqdm.auto import tqdm  # type: ignore
from transformers import (  # type: ignore
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.data import extract_boxed_answer  # For potential specific MATH eval
from src.data import extract_solution_marker_answer  # Primary answer extractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Placeholder for a more robust answer comparison function if needed
def compare_answers(predicted: Optional[str], expected: Optional[str]) -> bool:
    """Compare predicted and expected answers.

    Currently uses exact string matching after basic stripping.
    Expand for more robust checks (e.g., numerical equivalence).
    """
    if predicted is None or expected is None:
        return False
    return predicted.strip() == expected.strip()


def _generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    eval_batch_size: int,
    max_new_tokens: int,
    trace_handler: Optional[Callable],
) -> List[str]:
    """Helper function to generate model responses in batches."""
    model_generations = []
    logger.info(f"Generating responses with batch size {eval_batch_size}...")

    prof = None
    if trace_handler:
        prof = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(wait=0, warmup=0, active=1),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )
        prof.__enter__()

    batches_processed = 0
    for i in tqdm(range(0, len(prompts), eval_batch_size)):
        batch_prompts = prompts[i : i + eval_batch_size]
        _model_device = model.device  # type: ignore[attr-defined]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(_model_device)

        outputs = model.generate(  # type: ignore[attr-defined]
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,  # Greedy decoding for consistency
        )

        batch_generations = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        model_generations.extend(batch_generations)

        batches_processed += 1
        if prof:
            prof.step()
            if batches_processed >= 1:
                prof.__exit__(None, None, None)
                prof = None

    if (
        prof
    ):  # Ensure profiler exits if loop finishes early or wasn't entered enough
        prof.__exit__(None, None, None)
        prof = None

    return model_generations


def _determine_predicted_answer(
    generation: str,
    evaluation_type: str,
    dataset_name_for_logging: str,
    markers: Optional[Dict[str, str]],
) -> Optional[str]:
    """Determines the predicted answer based on evaluation strategy."""
    if evaluation_type == "standard":
        if dataset_name_for_logging == "math":
            # Standard MATH evaluation often relies on a specific format
            return extract_boxed_answer(generation)
        else:
            # For other standard evaluations, the gen itself is the answer.
            return generation.strip()
    elif evaluation_type == "reasoning":
        # For MATH reasoning, attempt marker extraction,
        # fallback to boxed if markers are problematic.
        if dataset_name_for_logging == "math":
            if markers and markers.get("solution_start"):
                return extract_solution_marker_answer(generation, markers)
            else:
                logger.info(
                    f"MATH '{dataset_name_for_logging}': bad markers, using "
                    "boxed."
                )
                return extract_boxed_answer(generation)
        else:  # Non-MATH reasoning
            if markers and markers.get("solution_start"):
                return extract_solution_marker_answer(generation, markers)
            else:
                logger.warning(
                    f"Reasoning '{dataset_name_for_logging}': bad markers,"
                    "cannot extract."
                )
                return None
    else:
        logger.error(
            f"Unknown evaluation_type '{evaluation_type}' for "
            f"{dataset_name_for_logging}."
        )
        return None


def _extract_and_log_results(
    prompts: List[str],
    generations: List[str],
    ground_truths: List[str],
    markers: Dict[str, str] | None,
    dataset_name_for_logging: str,  # To decide specific extraction if needed
    evaluation_type: str = "reasoning",
) -> tuple[int, List[Dict[str, Any]]]:
    """Extract answers, compare with ground truth, log, and count correct.

    Args:
        prompts: List of input prompts.
        generations: Model generations corresponding to prompts.
        ground_truths: Expected answers for each prompt.
        markers: Markers used for reasoning extraction. Can be ``None`` when
            ``evaluation_type`` is ``"standard"`` (non-reasoning mode).
        dataset_name_for_logging: Human-readable dataset identifier for logs.
        evaluation_type: Either ``"reasoning"`` or ``"standard"``
            (non-reasoning).

    Returns:
        Tuple containing the number of correct predictions and a list with per
        example logs.
    """
    correct_count = 0
    results_log = []  # Stores detailed log for each example
    logger.info("Extracting answers and calculating accuracy...")

    for i, (prompt, gen, expected_answer) in enumerate(
        zip(prompts, generations, ground_truths)
    ):
        predicted_answer = _determine_predicted_answer(
            gen, evaluation_type, dataset_name_for_logging, markers
        )

        is_correct = compare_answers(predicted_answer, expected_answer)
        if is_correct:
            correct_count += 1

        current_result = {
            "prompt": prompt,
            "generation": gen,
            "predicted_answer": predicted_answer,
            "expected_answer": expected_answer,
            "correct": is_correct,
        }
        results_log.append(current_result)

        # Log details for the first 10 examples
        if i < 10:
            logger.info("-" * 30)
            log_msg = f"Evaluation Example {i+1} ({dataset_name_for_logging}):"
            logger.info(log_msg)
            prompt_display = prompt
            if len(prompt) > 500:
                prompt_display = prompt[:497] + "..."
            logger.info(f"  Input Prompt:\n{prompt_display}\n")
            logger.info(f"  Model Generation:\n{gen}\n")
            logger.info(f"  Extracted Answer: {predicted_answer}")
            logger.info(f"  Ground Truth:     {expected_answer}")
            logger.info(f"  Correct:          {is_correct}")
            logger.info("-" * 30)

    return correct_count, results_log


@torch.inference_mode()
def evaluate_model(  # noqa: C901
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_dataset: Any,  # Changed: This is the pre-formatted HF dataset
    dataset_name_for_logging: str,  # Changed: For logging
    markers: Dict[str, str] | None = None,
    evaluation_type: str = "reasoning",
    max_new_tokens: int = 256,
    eval_batch_size: int = 32,
    num_samples: Optional[int] = None,  # Subsamples passed dataset
    trace_handler: Optional[Callable] = None,
) -> float:
    """
    Evaluate the model on a given pre-formatted dataset.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        eval_dataset: Pre-formatted Hugging Face dataset.
        dataset_name_for_logging: Original name of dataset for logging.
        markers: Markers required when ``evaluation_type`` is ``"reasoning"``.
            Can be ``None`` for the ``"standard"`` evaluation type.
        evaluation_type: One of ``"reasoning"`` (default) or
            ``"standard"``. Determines how prompts are crafted and answers
            are extracted.
        max_new_tokens: Maximum number of tokens for the answer.
        eval_batch_size: Batch size for evaluation inference.
        num_samples: Samples to evaluate from the provided dataset.
                     If None, use all samples in eval_dataset.
        trace_handler: Optional PyTorch profiler trace handler.

    Returns:
        Accuracy score.
    """
    logger.info(f"Starting evaluation on {dataset_name_for_logging}...")

    # Subsample if requested from the already formatted dataset
    if (
        num_samples is not None
        and num_samples > 0
        and num_samples < len(eval_dataset)
    ):
        logger.info(
            f"Subsampling to {num_samples} samples from the provided dataset."
        )
        eval_dataset = eval_dataset.select(range(num_samples))

    logger.info(
        f"Evaluating on {len(eval_dataset)} samples from "
        f"{dataset_name_for_logging}."
    )

    # Prompts are already formatted with system message by the data processor.
    # We just need to apply the chat template for final model input string.
    try:
        prompts: List[str] = []  # Explicitly type prompts
        for example in eval_dataset:
            applied_template = tokenizer.apply_chat_template(
                example["prompt"], tokenize=False, add_generation_prompt=True
            )
            assert isinstance(
                applied_template, str
            )  # Ensure it's a string for mypy
            prompts.append(applied_template)
        ground_truths = eval_dataset["answer"]
    except KeyError as e:
        raise ValueError(
            f"Dataset {dataset_name_for_logging} missing required columns "
            f"'prompt' or 'answer'. Error: {e}"
        )
    except TypeError as e:
        # Handle cases where eval_dataset can be an empty list or non-iterable
        if not eval_dataset:
            logger.warning(
                f"Received empty or invalid dataset for "
                f"{dataset_name_for_logging}. Skipping evaluation."
            )
            return 0.0
        raise ValueError(
            f"Error processing prompts/answers from eval_dataset for "
            f"{dataset_name_for_logging}: {e}"
        )

    if not prompts:
        logger.warning(
            f"No prompts generated for {dataset_name_for_logging}. "
            f"Skipping evaluation."
        )
        return 0.0

    model_generations = _generate_responses(
        model,
        tokenizer,
        prompts,
        eval_batch_size,
        max_new_tokens,
        trace_handler,
    )

    correct_count, results_log = _extract_and_log_results(
        prompts=prompts,
        generations=model_generations,
        ground_truths=ground_truths,
        markers=markers,
        dataset_name_for_logging=dataset_name_for_logging,
        evaluation_type=evaluation_type,
    )

    accuracy = (
        correct_count / len(model_generations)
        if len(model_generations) > 0
        else 0.0
    )
    logger.info(
        f"Final accuracy on {dataset_name_for_logging} \
        ({len(prompts)} samples): {accuracy:.4f}"
    )

    # Optional: Log detailed results (e.g., a few examples) to wandb
    # Example: wandb.log(
    #     {f"eval/{dataset_name_for_logging}_examples":
    #         wandb.Table(dataframe=pd.DataFrame(results_log[:20]))}
    # )
    # Note: For this to work, you'd need to pip install pandas and import it.

    return accuracy
