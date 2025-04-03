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
        default="/home/user/test_code_generation_models/outputs/checkpoint-50",
        help="Path to the trained model",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="If a train travels at 120 km/h and another train travels at "
        "180 km/h in the opposite direction, how long will it take for them to"
        " be 450 km apart if they start at the same location?",
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
