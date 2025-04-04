#!/usr/bin/env python3
"""
Entry point for training a model on the GSM8K dataset with GRPO.

This script uses Hydra for configuration management.
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from src.train import inference_demo, train


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training and optional inference function driven by Hydra config."""
    print(OmegaConf.to_yaml(cfg))

    # Run training
    # Note: The 'train' function needs to handle saving the model to
    # cfg.training.save_model_name within the Hydra run directory.
    # Hydra automatically changes the CWD to the run output directory.
    train(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        max_prompt_length=cfg.model.max_prompt_length,
        load_in_4bit=cfg.model.load_in_4bit,
        load_in_8bit=cfg.model.load_in_8bit,
        full_finetuning=cfg.training.full_finetuning,
        output_dir=cfg.training.output_dir,  # Pass base output dir if needed
        save_model_path=cfg.training.save_model_name,  # Now just the name
        eval_max_new_tokens=cfg.evaluation.max_new_tokens,
        profile=cfg.training.profile,
    )

    # Run inference if requested
    if cfg.inference.run:
        # Assuming inference_demo loads the model from the path saved by train
        # which should be cfg.training.save_model_name relative to CWD.
        inference_demo(
            model_path=cfg.training.save_model_name,
            query=cfg.inference.query,
        )


if __name__ == "__main__":
    main()
