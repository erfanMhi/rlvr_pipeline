import logging
from typing import Any, Dict, List, Optional, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers.trainer_callback import TrainerState

from src.components.data.interface import DataComponentInterface
from src.components.evaluation.interface import EvaluationComponentInterface
from src.components.model.interface import ModelComponentInterface
from src.components.observers.interface import ObserverInterface
from src.components.reward.interface import RewardComponentInterface
from src.components.training_loop.interface import TrainingLoopInterface

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the ML pipeline by managing and running components."""

    def __init__(self, pipeline_config: DictConfig):
        self.pipeline_config = pipeline_config
        self.observers: List[ObserverInterface] = []
        self.data_component: Optional[DataComponentInterface] = None
        self.model_component: Optional[ModelComponentInterface] = None
        self.reward_component: Optional[RewardComponentInterface] = None
        self.training_loop_component: Optional[TrainingLoopInterface] = None
        self.evaluation_component: Optional[EvaluationComponentInterface] = (
            None
        )
        self._setup_logging()
        self._init_observers()

    def _setup_logging(self) -> None:
        log_level_str = self.pipeline_config.get(
            "logging_level", "INFO"
        ).upper()
        level = getattr(logging, log_level_str, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.info(f"Logging setup with level {log_level_str}")

    def _init_observers(self) -> None:
        observer_configs = self.pipeline_config.get("observers", [])
        for obs_conf in observer_configs:
            try:
                # Create properly structured config for observer
                observer_config = self._create_component_config(obs_conf)
                observer = instantiate(observer_config)
                if observer is not None and observer.validate_config():
                    self.observers.append(observer)
                    logger.info(
                        f"Initialized observer: {type(observer).__name__}"
                    )
                else:
                    observer_name = (
                        type(observer).__name__ if observer else "None"
                    )
                    logger.error(
                        f"Observer config invalid or instantiation failed: "
                        f"{observer_name}"
                    )
            except Exception as e:
                msg = f"Failed to initialize observer: {e}"
                logger.error(msg, exc_info=True)
        logger.info(f"Initialized {len(self.observers)} observers.")

    def _notify_observers(self, event: str, *args: Any, **kwargs: Any) -> None:
        for observer in self.observers:
            try:
                method = getattr(observer, event, None)
                if method and callable(method):
                    method(*args, **kwargs)
            except Exception as e:
                msg = f"Error notifying observer for event {event}: {e}"
                logger.error(msg, exc_info=True)

    def _create_component_config(
        self, component_config: Union[DictConfig, Dict[str, Any]]
    ) -> DictConfig:
        """Create a properly structured config for component instantiation.

        Args:
            component_config: The original component configuration from
                pipeline_config

        Returns:
            DictConfig with _target_ and config keys properly structured
        """
        target = component_config["_target_"]
        # Create a copy to avoid modifying the original
        config_copy = DictConfig(component_config)
        del config_copy["_target_"]

        return DictConfig(
            {
                "_target_": target,
                "config": config_copy,
            }
        )

    def _init_data_component(self) -> None:
        if "data" not in self.pipeline_config:
            raise ValueError(
                "Data component configuration ('data') is missing "
                "in pipeline_config."
            )
        try:
            # Create properly structured config for data component
            data_config = self._create_component_config(
                self.pipeline_config.data
            )
            self.data_component = instantiate(data_config)
            if self.data_component is None:
                raise ValueError("DataComponent instantiation returned None")
            if not self.data_component.validate_config():
                raise ValueError("DataComponent configuration is invalid.")
            logger.info("DataComponent initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize DataComponent: {e}")
            raise

    def _init_model_component(self) -> None:
        if "model" not in self.pipeline_config:
            raise ValueError(
                "Model component configuration ('model') is missing "
                "in pipeline_config."
            )
        try:
            # Create properly structured config for model component
            resolved_config = OmegaConf.to_container(
                self.pipeline_config, resolve=True
            )  # need to resolve it beforehand because of the interpolations

            # Type safety check
            if not isinstance(resolved_config, dict):
                raise ValueError(
                    f"Expected resolved_config to be a dict, got "
                    f"{type(resolved_config)}"
                )

            if "model" not in resolved_config:
                raise ValueError("'model' key not found in resolved_config")

            model_config = self._create_component_config(
                resolved_config["model"]
            )
            self.model_component = instantiate(model_config)
            if self.model_component is None:
                raise ValueError("ModelComponent instantiation returned None")
            if not self.model_component.validate_config():
                raise ValueError("ModelComponent configuration is invalid.")
            logger.info("ModelComponent initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ModelComponent: {e}")
            raise

    def _init_reward_component(self) -> None:
        try:
            # Create properly structured config for reward component
            reward_config = self._create_component_config(
                self.pipeline_config.reward
            )
            self.reward_component = instantiate(reward_config)
            if self.reward_component is None:
                raise ValueError("RewardComponent instantiation returned None")
            if not self.reward_component.validate_config():
                raise ValueError("RewardComponent configuration is invalid.")
            logger.info("RewardComponent initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize RewardComponent: {e}")
            raise

    def _init_training_loop_component(self) -> None:
        if "train" not in self.pipeline_config:
            raise ValueError(
                "Training loop component configuration ('train') is missing "
                "in pipeline_config."
            )
        try:
            # For interpolation purposes we need to resolve it before creation
            resolved_config = OmegaConf.to_container(
                self.pipeline_config, resolve=True
            )  # need to resolve it beforehand because of the interpolations

            # Type safety check
            if not isinstance(resolved_config, dict):
                raise ValueError(
                    f"Expected resolved_config to be a dict, got "
                    f"{type(resolved_config)}"
                )

            if "train" not in resolved_config:
                raise ValueError("'train' key not found in resolved_config")

            training_config = self._create_component_config(
                resolved_config["train"]
            )
            self.training_loop_component = instantiate(training_config)
            if self.training_loop_component is None:
                raise ValueError(
                    "TrainingLoopComponent instantiation returned None"
                )
            if not self.training_loop_component.validate_config():
                raise ValueError(
                    "TrainingLoopComponent configuration is invalid."
                )
            logger.info("TrainingLoopComponent initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize TrainingLoopComponent: {e}")
            raise

    def _init_evaluation_component(self) -> None:
        eval_section_config = self.pipeline_config.get("eval")

        is_enabled = False
        if eval_section_config is not None:
            is_enabled = eval_section_config.get("enabled", False)

        if is_enabled:
            if eval_section_config is None:
                raise ValueError(
                    "Evaluation component configuration ('eval') is missing "
                    "in pipeline_config but component is enabled by default."
                )
            # Get the prompts configuration
            prompts_config = self.pipeline_config.get("prompts", {})
            if not prompts_config or not prompts_config.get("system_prompts"):
                raise ValueError(
                    "'prompts.system_prompts' configuration is missing or "
                    "empty, but is required for the EvaluationComponent."
                )

            try:
                # Create config with eval_config and prompts_config
                eval_component_config = self._create_component_config(
                    eval_section_config
                )
                # Add prompts_config to the restructured config
                eval_component_config["prompts_config"] = prompts_config
                self.evaluation_component = instantiate(eval_component_config)

                if self.evaluation_component is None:
                    raise ValueError(
                        "EvaluationComponent instantiation returned None"
                    )
                if not self.evaluation_component.validate_config():
                    raise ValueError(
                        "EvaluationComponent configuration is invalid."
                    )

                # Set dependencies if the component supports it
                if (
                    hasattr(self.evaluation_component, "set_dependencies")
                    and self.data_component
                    and self.model_component
                ):
                    # Type ignore: set_dependencies not in interface
                    self.evaluation_component.set_dependencies(  # type: ignore
                        data_comp=self.data_component,
                        model_comp=self.model_component,
                    )
                logger.info("EvaluationComponent initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize EvaluationComponent: {e}")
                raise
        else:
            logger.info("EvaluationComponent is disabled in config.")

    def run(self) -> None:
        """Runs the entire ML training pipeline."""
        try:
            self._notify_observers(
                "on_pipeline_start", orchestrator_config=self.pipeline_config
            )
            logger.info("Pipeline starting...")

            # 1. Initialize components
            self._notify_observers("on_step_start", step_name="init_data_comp")
            self._init_data_component()
            assert self.data_component, "Data component not initialized"
            self._notify_observers("on_step_end", step_name="init_data_comp")

            self._notify_observers(
                "on_step_start", step_name="init_model_comp"
            )
            self._init_model_component()
            assert self.model_component, "Model component not initialized"
            self._notify_observers("on_step_end", step_name="init_model_comp")

            self._notify_observers(
                "on_step_start", step_name="init_reward_comp"
            )
            self._init_reward_component()
            assert self.reward_component, "Reward component not initialized"
            self._notify_observers("on_step_end", step_name="init_reward_comp")

            self._notify_observers(
                "on_step_start", step_name="init_training_loop"
            )
            self._init_training_loop_component()
            assert self.training_loop_component, "Loop comp not initialized"
            self._notify_observers(
                "on_step_end", step_name="init_training_loop"
            )

            self._notify_observers(
                "on_step_start", step_name="init_evaluation_comp"
            )
            self._init_evaluation_component()
            self._notify_observers(
                "on_step_end", step_name="init_evaluation_comp"
            )

            # extract the system prompts
            prompts_config = self.pipeline_config["prompts"]
            system_prompts_dict = prompts_config.get("system_prompts")

            if (
                not isinstance(system_prompts_dict, (dict, DictConfig))
                or not system_prompts_dict
            ):
                raise ValueError(
                    "Configuration error: 'prompts.system_prompts' must be a "
                    "non-empty dictionary."
                )

            # Enables switching between reasoning and non-reasoning modes
            prompting_mode = prompts_config.get(
                "prompting_mode", "reasoning_mode"
            )

            system_prompt_str = system_prompts_dict.get(prompting_mode)
            if system_prompt_str is None:
                keys_available = list(system_prompts_dict.keys())
                error_msg = (
                    f"Config error: Prompt key '{prompting_mode}' from "
                    f"'prompts.prompting_mode' "
                    f"not in 'prompts.system_prompts'. Keys: {keys_available}"
                )
                raise ValueError(error_msg)

            # 2. Setup: Load data, model, tokenizer etc.
            logger.info("Setting up model and tokenizer...")
            self._notify_observers("on_step_start", step_name="model_setup")
            model, tokenizer = (
                self.model_component.initialize_model_and_tokenizer()
            )
            output = {
                "model_name": (
                    model.name_or_path
                    if hasattr(model, "name_or_path")
                    else "custom"
                )
            }
            self._notify_observers(
                "on_step_end", step_name="model_setup", output=output
            )

            logger.info("Loading and preparing data...")
            self._notify_observers(
                "on_step_start", step_name="data_preparation"
            )

            train_dataset = self.data_component.load_and_prepare_data(
                tokenizer=tokenizer,
                system_prompt=system_prompt_str,
                split="train",
            )
            self._notify_observers(
                "on_step_end",
                step_name="data_preparation",
                output={"dataset_size": len(train_dataset)},
            )

            logger.info("Preparing reward functions...")
            model_info = {"markers": self.model_component.get_markers()}
            self._notify_observers("on_step_start", step_name="reward_setup")
            reward_functions = self.reward_component.get_reward_pipelines(
                model_info=model_info,
            )
            if not reward_functions:
                # TODO: this should be modified later
                raise ValueError("No reward functions found.")
            self._notify_observers("on_step_end", step_name="reward_setup")

            # 3. Setup Callbacks
            trainer_callbacks = []
            if self.evaluation_component:
                self._notify_observers(
                    "on_step_start", step_name="eval_cb_setup"
                )
                eval_callback = self.evaluation_component.get_trainer_callback(
                    data_component_instance=self.data_component,
                    model_component_instance=self.model_component,
                    tokenizer=tokenizer,
                )
                if eval_callback:
                    trainer_callbacks.append(eval_callback)
                self._notify_observers(
                    "on_step_end", step_name="eval_cb_setup"
                )

            # 4. Training
            logger.info("Starting training loop...")
            self._notify_observers("on_step_start", step_name="training_run")
            self.training_loop_component.train(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                reward_functions=reward_functions,
                callbacks=trainer_callbacks,
            )
            self._notify_observers("on_step_end", step_name="training_run")

            # 5. Post-training evaluation
            if self.evaluation_component:
                logger.info("Starting post-training evaluation...")
                self._notify_observers(
                    "on_step_start", step_name="post_train_eval"
                )

                mock_state = TrainerState()
                self.evaluation_component.on_evaluation_run(
                    model=model,
                    tokenizer=tokenizer,
                    trainer_state=mock_state,
                )
                self._notify_observers(
                    "on_step_end", step_name="post_train_eval"
                )

            # 6. Save final model
            model_config = self.model_component.config
            save_path_cfg = model_config.get("save_model_path")  # type: ignore
            if save_path_cfg:
                logger.info(f"Saving final model to {save_path_cfg}...")
                self._notify_observers(
                    "on_step_start", step_name="save_final_model"
                )
                self.model_component.save_model(
                    model, tokenizer, save_path_cfg
                )  # type: ignore
                self._notify_observers(
                    "on_step_end",
                    step_name="save_final_model",
                    output={"path": save_path_cfg},
                )

            self._notify_observers("on_pipeline_end", status="SUCCESS")
            logger.info("Pipeline finished successfully.")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self._notify_observers(
                "on_pipeline_end", status="FAILURE", error=e
            )
            raise
