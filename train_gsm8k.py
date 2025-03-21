#!/usr/bin/env python3
"""
Entry point for training a model on the GSM8K dataset with GRPO.

This script provides a command-line interface to run training with
configurable parameters.
"""

import argparse

from src.train import inference_demo, train


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a model on GSM8K using GRPO"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/gemma-3-1b-it",
        help="Name or path of the model to fine-tune",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )

    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=256,
        help="Maximum prompt length",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )

    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )

    parser.add_argument(
        "--full_finetuning",
        action="store_true",
        help="Perform full model finetuning",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save training outputs",
    )

    parser.add_argument(
        "--save_model_path",
        type=str,
        default="gemma-3",
        help="Path to save the final model",
    )

    parser.add_argument(
        "--run_inference",
        action="store_true",
        help="Run inference demo after training",
    )

    parser.add_argument(
        "--inference_query",
        type=str,
        default="What is the sqrt of 101?",
        help="Query to test during inference",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Run training
    train(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
        output_dir=args.output_dir,
        save_model_path=args.save_model_path,
    )

    # Run inference if requested
    if args.run_inference:
        inference_demo(
            model_path=args.save_model_path,
            query=args.inference_query,
        )
