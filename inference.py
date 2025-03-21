#!/usr/bin/env python3
"""Run inference using a trained GSM8K model."""

import argparse

from src.train import inference_demo


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained GSM8K model"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="gemma-3",
        help="Path to the trained model",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="What is the sqrt of 101?",
        help="The math problem to solve",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Run inference
    inference_demo(
        model_path=args.model_path,
        query=args.query,
        max_new_tokens=args.max_new_tokens,
    )
