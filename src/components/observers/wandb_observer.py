import logging
import os
from typing import Any, Dict, List, Optional

import wandb

from src.components.observers.interface import ObserverInterface

logger = logging.getLogger(__name__)


class WandbObserver(ObserverInterface):
    """Observer for logging pipeline events and metrics to Weights & Biases."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.wandb_project = self.config.get(
            "wandb_project", "default_grpo_project"
        )
        self.wandb_entity = self.config.get("wandb_entity")
        self.wandb_run_name = self.config.get("wandb_run_name")
        self.log_config = self.config.get("log_config_to_wandb", True)
        self.pipeline_config_logged = False

    def validate_config(self) -> bool:
        if not self.wandb_project:
            logger.error(
                "WandbObserver config must include 'wandb_project' name."
            )
            return False
        # wandb.login() should have been called externally if API key needed.
        return True

    def on_pipeline_start(self, orchestrator_config: Dict[str, Any]) -> None:
        """Initialize wandb run."""
        if not self.validate_config():
            logger.error(
                "WandbObserver configuration invalid. Cannot start wandb run."
            )
            return

        try:
            if wandb.run is not None:
                logger.warning(
                    f"Existing wandb run (id: {wandb.run.id}) detected. "
                    "This observer will use the existing run. If unexpected, "
                    "ensure wandb.finish() was called."
                )
            else:
                run_info_parts = [
                    "Initializing wandb run for project "
                    f"'{self.wandb_project}'"
                ]
                if self.wandb_entity:
                    run_info_parts.append(f", entity '{self.wandb_entity}'")
                if self.wandb_run_name:
                    run_info_parts.append(
                        f", run_name '{self.wandb_run_name}'"
                    )
                logger.info("".join(run_info_parts))

                wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=self.wandb_run_name,
                    config=(
                        orchestrator_config if self.log_config else None
                    ),  # Log the full pipeline config
                    # Allow reinit if run exists but wandb.run is None
                    reinit=True,
                )
            self.pipeline_config_logged = bool(
                self.log_config and orchestrator_config
            )
            url = wandb.run.url if wandb.run else "N/A"
            logger.info(f"Wandb run started. URL: {url}")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}", exc_info=True)

    def on_pipeline_end(
        self, status: str, error: Optional[Exception] = None
    ) -> None:
        """Finish wandb run and log final status."""
        if wandb.run:
            logger.info(
                f"Pipeline ended with status: {status}. Finishing wandb run."
            )
            if error:
                wandb.log({"pipeline_error": str(error)})
                wandb.summary["pipeline_status"] = f"ERROR: {status}"
            else:
                wandb.summary["pipeline_status"] = status
            wandb.finish(
                exit_code=0 if status == "SUCCESS" and not error else 1
            )
        else:
            logger.warning(
                "on_pipeline_end called but no active wandb run found."
            )

    def on_step_start(
        self, step_name: str, step_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log the start of a pipeline step."""
        if wandb.run:
            # Detailed per-step logging is usually handled by GRPOTrainer
            # callbacks or the specific component's logic.
            logger.debug(f"WandbObserver: Step '{step_name}' started.")
        pass

    def on_step_end(
        self,
        step_name: str,
        output: Optional[Any] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Log the end of a pipeline step, possibly with output or error."""
        if wandb.run:
            if error:
                logger.error(
                    f"WandbObserver: Step '{step_name}' "
                    f"ended with error: {error}"
                )
            else:
                logger.debug(f"WandbObserver: Step '{step_name}' ended.")
                # Output logging can be added if needed, e.g.:
                # if isinstance(output, dict):
                #     wandb.log({f"step_output/{step_name}/{k}": v
                #                  for k, v in output.items()})
        pass

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "custom",
    ) -> None:
        """Logs a dictionary of metrics to wandb at a specific step."""
        if wandb.run:
            log_payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
            if step is not None:
                wandb.log(log_payload, step=step)
            else:
                wandb.log(log_payload)
            logger.debug(
                f"Logged custom metrics to wandb with prefix '{prefix}': "
                f"{metrics}"
            )

    def log_completions_table(
        self,
        table_name: str,
        columns: List[str],
        data: List[List[Any]],
        step: Optional[int] = None,
    ) -> None:
        """Logs a table of completions (or other tabular data) to wandb."""
        if wandb.run:
            table = wandb.Table(columns=columns, data=data)
            log_entry = {table_name: table}
            if step is not None:
                wandb.log(log_entry, step=step)
            else:
                wandb.log(log_entry)
            logger.debug(f"Logged table '{table_name}' to wandb.")

    def log_artifact(
        self, artifact_path: str, artifact_name: str, artifact_type: str
    ) -> None:
        """Logs an artifact (e.g., model, dataset) to wandb."""
        if wandb.run:
            try:
                artifact = wandb.Artifact(
                    name=artifact_name, type=artifact_type
                )
                if os.path.isfile(artifact_path):
                    artifact.add_file(artifact_path)
                elif os.path.isdir(artifact_path):
                    artifact.add_dir(artifact_path)
                else:
                    logger.error(
                        f"Wandb artifact path not found: {artifact_path}"
                    )
                    return
                wandb.log_artifact(artifact)
                logger.info(
                    f"Logged artifact '{artifact_name}' of type "
                    f"'{artifact_type}' from {artifact_path}."
                )
            except Exception as e:
                logger.error(
                    f"Failed to log wandb artifact {artifact_name}: {e}",
                    exc_info=True,
                )
