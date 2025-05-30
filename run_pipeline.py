#!/usr/bin/env python3
"""Pipeline Orchestration Entry Point.

This script serves as the main entry point for running the ML pipeline using
Hydra for configuration management and the PipelineOrchestrator for
execution coordination.

Usage:
    python run_pipeline.py
    python run_pipeline.py data_component.max_train_samples=500
    python run_pipeline.py --config-name=custom_config
    python run_pipeline.py --multirun \
        data_component.max_train_samples=100,200,500

The pipeline configuration is defined in the `conf` directory and follows the
structure established by `conf/config.yaml` and its component config groups.
"""
# Imported in the beginning to make sure it's imported before all modules
from unsloth import FastLanguageModel  # isort: skip  # noqa: E402,F401,I100

import builtins  # noqa: I100
import logging
import math
import os
import sys
from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

# Import our pipeline orchestrator
from src.orchestration.pipeline_orchestrator import PipelineOrchestrator
from src.utils.custom_hydra_resolvers import register_custom_resolvers

# Register custom OmegaConf resolvers
# Must be done BEFORE @hydra.main decorator
register_custom_resolvers()  # For ${get_marker:...}

# Safest: expose only what you need for the existing "eval" resolver
_allowed_builtins = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if not name.startswith("_")
}
_allowed_eval_globals = {"math": math, "__builtins__": _allowed_builtins}
OmegaConf.register_new_resolver(
    "eval", lambda expr: eval(expr, _allowed_eval_globals)
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_hydra_config(cfg: DictConfig) -> bool:
    """Validate that all required components have proper _target_ keys.

    Args:
        cfg: The Hydra configuration object

    Returns:
        True if configuration is valid, False otherwise
    """
    required_components = ["data", "model", "reward", "train"]

    for component in required_components:
        if component not in cfg:
            logger.error(f"Missing {component} component configuration")
            return False

        if "_target_" not in cfg[component]:
            logger.error(f"Missing _target_ key in {component} configuration")
            return False

    # Validate observers if present
    observers = cfg.get("observers", [])
    for i, observer in enumerate(observers):
        if "_target_" not in observer:
            logger.error(
                f"Missing _target_ key in observer configuration at index {i}"
            )
            return False

    logger.info("Configuration validation passed")
    return True


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> int:
    """Main entry point for the ML pipeline."""
    # Print config summary (useful for debugging)
    if cfg.get("verbose_config", False):
        logger.info("Pipeline Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))
    else:
        logger.info("Starting pipeline with Hydra configuration")

    # Validate configuration structure
    if not validate_hydra_config(cfg):
        logger.error("Configuration validation failed. Exiting.")
        return 1

    # Log paths for clarity - helps troubleshoot file paths during execution
    logger.info(f"Project root: {hydra.utils.get_original_cwd()}")
    logger.info(f"Hydra output dir: {os.getcwd()}")

    # Track whether the pipeline was successful (for return code)
    pipeline_success = False

    # Using Dict[str, Any] for broader compatibility
    resolved_config: Any = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_config, dict):
        raise TypeError(
            f"Expected configuration to be a dict, got {type(resolved_config)}"
        )
    config: Dict[str, Any] = resolved_config

    try:
        # Instantiate and run the pipeline orchestrator
        logger.info("Initializing PipelineOrchestrator...")
        orchestrator = PipelineOrchestrator(config)

        logger.info("Starting pipeline execution...")
        orchestrator.run()

        logger.info("Pipeline execution completed successfully!")
        pipeline_success = True

        # Run inference if configured
        if cfg.get("inference", {}).get("run", False):
            logger.info("Running post-training inference...")
            # run_inference(cfg)

        return 0  # Success

    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user.")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1  # Failure

    finally:
        # Summary message
        if pipeline_success:
            logger.info("✅ Pipeline completed successfully")
        else:
            logger.warning("❌ Pipeline did not complete successfully")

        # Log the location of Hydra's output directory
        if hydra.core.hydra_config.HydraConfig.initialized():
            output_dir = (
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
            logger.info(f"Hydra output directory: {output_dir}")
            # Optionally create a symlink to the latest run for convenience
            try:
                latest_link = os.path.join(
                    hydra.utils.get_original_cwd(), "outputs/latest"
                )
                if os.path.exists(latest_link):
                    os.remove(latest_link)
                os.makedirs(os.path.dirname(latest_link), exist_ok=True)
                os.symlink(output_dir, latest_link)
                logger.info(f"Created symlink to latest run: {latest_link}")
            except Exception as e:
                logger.debug(f"Could not create latest run symlink: {e}")


if __name__ == "__main__":
    # Proper exit handling for command line usage
    sys.exit(main())
