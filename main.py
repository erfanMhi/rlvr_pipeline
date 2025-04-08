#!/usr/bin/env python3
"""
Entry point for training a model on the GSM8K dataset with GRPO.

This script uses Hydra for configuration management.
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.train import inference_demo, train

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training and optional inference function driven by Hydra config."""
    logger.info(OmegaConf.to_yaml(cfg))

    train(
        # Model parameters
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        max_prompt_length=cfg.model.max_prompt_length,
        load_in_4bit=cfg.model.load_in_4bit,
        load_in_8bit=cfg.model.load_in_8bit,
        # Data parameters
        training_dataset_name=cfg.data.train_dataset_name,
        eval_datasets=list(cfg.data.eval_datasets),  # Ensure it's a list
        # Optimizer parameters
        optim_name=cfg.optimizer.name,
        learning_rate=cfg.optimizer.learning_rate,
        adam_beta1=cfg.optimizer.adam_beta1,
        adam_beta2=cfg.optimizer.adam_beta2,
        weight_decay=cfg.optimizer.weight_decay,
        # Training parameters
        full_finetuning=cfg.training.full_finetuning,
        trainer_output_dir=cfg.training.trainer_internal_output_dir,
        save_model_path=cfg.training.save_model_name,  # For final model save
        profile=cfg.training.profile,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        logging_steps=cfg.training.logging_steps,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_steps=cfg.training.max_steps,
        save_steps=cfg.training.save_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        report_to=list(cfg.training.report_to),  # Ensure it's a list
        log_completions_to_wandb=cfg.training.log_completions_to_wandb,
        # GRPO parameters
        grpo_num_generations=cfg.grpo.num_generations,
        # Evaluation parameters
        evaluation_enabled=cfg.evaluation.enabled,
        eval_steps=cfg.evaluation.steps,
        eval_num_samples=(
            cfg.evaluation.num_samples
            if cfg.evaluation.num_samples is not None
            else None
        ),
        eval_batch_size=cfg.evaluation.batch_size,
        eval_max_new_tokens=cfg.evaluation.max_new_tokens,
        # Wandb parameters
        wandb_project=cfg.wandb.project,
    )

    # Run inference if requested
    if cfg.inference.run:
        inference_demo(
            # Assumes model saved relative to CWD
            model_path=cfg.training.save_model_name,
            query=cfg.inference.query,
            # Assumes model saved relative to CWD
            max_new_tokens=cfg.evaluation.max_new_tokens,
        )


if __name__ == "__main__":
    main()
