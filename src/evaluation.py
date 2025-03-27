"""Model evaluation script."""

# import os # Unused
from typing import Callable, Dict, Optional  # Removed Any, List

import torch

# import wandb # Unused
import torch.profiler as profiler  # Import profiler

# from datasets import Dataset  # type: ignore # Unused
from tqdm.auto import tqdm  # type: ignore
from transformers import (  # type: ignore
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.data import (  # extract_hash_answer, # Unused
    create_prompt_format,
    extract_boxed_answer,
    extract_solution_marker_answer,
    load_gsm8k_dataset,
    load_math_dataset,
    load_svamp_dataset,
)


# Placeholder for a more robust answer comparison function if needed
def compare_answers(predicted: Optional[str], expected: Optional[str]) -> bool:
    """Compare predicted and expected answers.

    Currently uses exact string matching after basic stripping.
    Expand this function for more robust checks (e.g., numerical equivalence).
    """
    if predicted is None or expected is None:
        return False
    return predicted.strip() == expected.strip()


@torch.inference_mode()
def evaluate_model(  # noqa: C901
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    system_prompt: str,
    markers: Dict[str, str],
    max_new_tokens: int = 256,
    eval_batch_size: int = 32,
    num_samples: Optional[int] = None,
    trace_handler: Optional[Callable] = None,  # Use Callable type
) -> float:
    """
    Evaluate the model on a given dataset (GSM8K or MATH).

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        dataset_name: Name of the dataset ('gsm8k' or 'math').
        system_prompt: System prompt for formatting.
        markers: Reasoning markers for formatting.
        max_new_tokens: Maximum number of tokens for the answer.
        eval_batch_size: Batch size for evaluation inference.
        num_samples: Number of samples to evaluate on. If None, use all.
        trace_handler: An optional configured PyTorch profiler trace handler.

    Returns:
        Accuracy score.
    """
    print(f"Starting evaluation on {dataset_name}...")

    # Load dataset
    if dataset_name == "gsm8k":
        eval_dataset = load_gsm8k_dataset(split="test")
    elif dataset_name == "math":
        eval_dataset = load_math_dataset(split="test")
    elif dataset_name == "svamp":
        eval_dataset = load_svamp_dataset(split="test")
    else:
        raise ValueError(f"Unknown dataset for evaluation: {dataset_name}")

    # Subsample if requested
    if num_samples is not None and num_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(num_samples))
        print(f"Evaluating on {num_samples} samples.")
    else:
        print(f"Evaluating on all {len(eval_dataset)} samples.")

    # Format dataset
    # Note: Markers aren't strictly needed if system prompt guides format.
    # Primary goal is correct input prompt formatting.
    formatted_dataset = create_prompt_format(
        system_prompt, markers, eval_dataset
    )

    prompts = [
        tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )
        for example in formatted_dataset
    ]
    ground_truths = formatted_dataset["answer"]  # Already extracted

    model_generations = []
    print(f"Generating responses with batch size {eval_batch_size}...")

    # --- Profiling Setup --- #
    prof = None
    if trace_handler is not None:  # Check if handler was provided
        prof = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(
                wait=0, warmup=0, active=1
            ),  # Profile only 1 batch
            on_trace_ready=trace_handler,  # Use the provided handler
            record_shapes=True,
            profile_memory=False,  # Disable memory profiling
            with_stack=False,  # Already disabled
        )
        prof.__enter__()  # Manually enter context
    # --- End Profiling Setup --- #

    batches_processed = 0
    for i in tqdm(range(0, len(prompts), eval_batch_size)):
        batch_prompts = prompts[i : i + eval_batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,  # Use greedy decoding for evaluation consistency
        )

        # Decode generated tokens, skipping prompt
        batch_generations = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        model_generations.extend(batch_generations)

        batches_processed += 1

        # --- Profiling Step/Exit --- #
        if prof:
            prof.step()  # Signal profiler step
            if batches_processed >= 1:  # Exit after profiling 1 batch
                prof.__exit__(None, None, None)  # Manually exit context
                prof = None  # Prevent further stepping/exiting
        # --- End Profiling Step/Exit --- #

    # --- Profiling Exit (if loop finished before 1 batch) --- #
    if prof:
        prof.__exit__(None, None, None)
        prof = None
    # --- End Profiling Exit --- #

    # Extract answers and calculate accuracy
    correct_count = 0
    results = []
    print("Calculating accuracy...")
    for i, (gen, expected_answer) in enumerate(
        zip(model_generations, ground_truths)
    ):
        # --- Call specific extractor based on dataset ---
        predicted_answer: Optional[str] = None
        if dataset_name in ["gsm8k", "svamp"]:
            predicted_answer = extract_solution_marker_answer(gen, markers)
        elif dataset_name == "math":
            predicted_answer = extract_boxed_answer(gen)
        # --- End specific extractor call ---

        is_correct = compare_answers(predicted_answer, expected_answer)
        if is_correct:
            correct_count += 1

        # Log details for the first 10 examples
        if i < 10:
            print("-" * 30)
            print(f"Evaluation Example {i+1}:")
            # Truncate long prompts for readability
            prompt_display = (
                (prompts[i][:500] + "...")
                if len(prompts[i]) > 500
                else prompts[i]
            )
            print(f"  Input Prompt:\n{prompt_display}\n")
            print(f"  Model Generation:\n{gen}\n")
            print(f"  Extracted Answer: {predicted_answer}")
            print(f"  Ground Truth:     {expected_answer}")
            print(f"  Correct:          {is_correct}")
            print("-" * 30)

        results.append(
            {
                "prompt": prompts[i],
                "generation": gen,
                "predicted_answer": predicted_answer,
                "expected_answer": expected_answer,
                "correct": is_correct,
            }
        )

    accuracy = (
        correct_count / len(model_generations)
        if len(model_generations) > 0
        else 0.0
    )
    print(f"Evaluation finished. Accuracy on {dataset_name}: {accuracy:.4f}")

    # Optional: Log detailed results (e.g., a few examples) to wandb
    # Example: wandb.log({f"eval/{dataset_name}_examples":
    #                     wandb.Table(dataframe=pd.DataFrame(results[:20]))})

    return accuracy
