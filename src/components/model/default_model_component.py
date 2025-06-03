# Import unsloth before transformers for performance optimizations
from unsloth import (  # isort: skip  # noqa: E402,I100,I201
    FastLanguageModel,  # Assuming unsloth is the primary way to load models
)

import logging  # noqa: E402,I100,I201
from typing import Any, Dict, Optional, Tuple, cast

from omegaconf import DictConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.components.model.interface import ModelComponentInterface

# Config related functions will be moved/adapted from src.config
# Model loading/saving functions from src.model

logger = logging.getLogger(__name__)


def _get_model_config_dict(
    model_name_key: str, component_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Retrieves model-specific config from the component_config.

    It expects 'available_model_specific_configs' in component_config.
    """
    available_configs = component_config.get(
        "available_model_specific_configs", {}
    )
    # Ensure available_configs is treated as a dictionary
    if not isinstance(available_configs, dict):
        logger.warning(
            "'available_model_specific_configs' is not a dictionary. "
            "Returning empty model config."
        )
        return {}

    model_conf = available_configs.get(model_name_key, {})
    if not isinstance(model_conf, dict):
        logger.warning(
            f"Configuration for model key '{model_name_key}' is not a dict. "
            "Returning empty model config."
        )
        return {}

    if not model_conf:  # Log if empty after checks
        logger.warning(
            f"No config for model '{model_name_key}' in "
            f"'available_model_specific_configs'."
        )
    return model_conf


# --- DefaultModelComponent Implementation ---
class DefaultModelComponent(ModelComponentInterface):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        model_name_or_path_val = self.config.get("model_name_or_path")
        if (
            not isinstance(model_name_or_path_val, str)
            or not model_name_or_path_val
        ):
            raise ValueError(
                "ModelComponent config must include a non-empty "
                "'model_name_or_path' string"
            )
        self.model_name: str = model_name_or_path_val

        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None

        self._model_params = _get_model_config_dict(
            self.model_name, self.config
        )
        if not self._model_params:  # Check if it's empty
            logger.warning(
                f"No specific params for model {self.model_name} "
                f"in 'available_model_specific_configs'. "
                f"Markers might be unavailable."
            )
            self._model_params = {}  # Ensure it's a dict

    def validate_config(self) -> bool:
        # model_name_or_path already validated in __init__
        if not self.config.get("max_seq_length"):
            logger.error(
                "'max_seq_length' not specified for model initialization."
            )
            return False

        lora_config = self.config.get("lora_config", {})
        if not isinstance(lora_config, (dict, DictConfig)):
            logger.error("'lora_config' must be a dictionary or DictConfig.")
            return False

        if not self.config.get("full_finetuning", False) and lora_config.get(
            "use_lora", False
        ):
            r_value = lora_config.get("r")
            lora_alpha_value = lora_config.get("lora_alpha")

            if not isinstance(r_value, int):
                logger.error(
                    "If using LoRA, 'r' must be an integer in 'lora_config'."
                )
                return False

            if not isinstance(lora_alpha_value, (int, float)):
                logger.error(
                    "If using LoRA, 'lora_alpha' must be an integer or float "
                    "in 'lora_config'."
                )
                return False

        # Validate markers if present
        markers = self.config.get("markers", {})
        if markers and not isinstance(markers, (dict, DictConfig)):
            logger.error("'markers' must be a dictionary or DictConfig.")
            return False

        # Validate available_model_specific_configs if present
        available_configs = self.config.get(
            "available_model_specific_configs", {}
        )
        if available_configs and not isinstance(
            available_configs, (dict, DictConfig)
        ):
            logger.error(
                "'available_model_specific_configs' must be a dictionary or "
                "DictConfig."
            )
            return False

        return True

    def initialize_model_and_tokenizer(
        self,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if self._model is not None and self._tokenizer is not None:
            logger.info(
                f"Model {self.model_name} and tokenizer already initialized."
            )
            return self._model, self._tokenizer

        logger.info(f"Initializing model: {self.model_name}")
        max_seq_length = self.config["max_seq_length"]
        load_in_4bit = self.config.get("load_in_4bit", False)
        load_in_8bit = self.config.get("load_in_8bit", False)
        fast_inference = self.config.get("fast_inference", True)
        attn_implementation = self.config.get("attn_implementation", "sdpa")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            fast_inference=fast_inference,
            attn_implementation=attn_implementation,
            full_finetuning=self.config.get("full_finetuning", False),
        )

        self._model = model
        self._tokenizer = tokenizer

        if not self.config.get("full_finetuning", False):
            lora_config = self.config.get("lora_config", {})
            if isinstance(lora_config, dict) and lora_config.get(
                "use_lora", False
            ):
                if self._model is None:  # Should not happen here
                    raise RuntimeError("Model is None before adding adapters.")
                self._model = self.add_adapters(self._model)
            else:
                logger.info(
                    "LoRA not enabled or lora_config missing/invalid. "
                    "Skipping adapter addition."
                )
        else:
            logger.info(
                "Full finetuning enabled. Skipping LoRA adapter addition."
            )

        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Model or tokenizer failed to initialize properly."
            )
        logger.info(
            f"Model {self.model_name} and tokenizer initialized successfully."
        )
        return self._model, self._tokenizer

    def add_adapters(self, model: PreTrainedModel) -> PreTrainedModel:
        lora_config = self.config.get("lora_config", {})
        if not isinstance(lora_config, dict) or not lora_config.get(
            "use_lora", False
        ):
            logger.info(
                "lora_config.use_lora is false or config invalid. "
                "Skipping adapter addition."
            )
            return model

        logger.info(f"Adding LoRA adapters to {self.model_name}.")
        r = lora_config.get("r")
        lora_alpha = lora_config.get("lora_alpha")

        if not isinstance(r, int) or not isinstance(lora_alpha, (int, float)):
            logger.error(
                "LoRA 'r' (int) and 'lora_alpha' (int/float) must be "
                "specified and be of correct types in 'lora_config' "
                "when use_lora is true."
            )
            return model  # Return original model if config is invalid

        target_modules = lora_config.get("target_modules")

        # Cast is used here as PeftModel is a PreTrainedModel.
        # Mypy might not infer this perfectly from library stubs.
        adapted_model = cast(
            PreTrainedModel,
            FastLanguageModel.get_peft_model(
                model,
                r=r,
                lora_alpha=int(lora_alpha),  # Ensure it's an int
                target_modules=target_modules,
                lora_dropout=lora_config.get("lora_dropout", 0.0),
                bias=lora_config.get("lora_bias", "none"),
                use_gradient_checkpointing=lora_config.get(
                    "use_gradient_checkpointing", "unsloth"
                ),
                random_state=lora_config.get("random_state", 3407),
                max_seq_length=self.config.get("max_seq_length", 1536),
                finetune_vision_layers=lora_config.get(
                    "finetune_vision_layers", False
                ),
                finetune_language_layers=lora_config.get(
                    "finetune_language_layers", True
                ),
                finetune_attention_modules=lora_config.get(
                    "finetune_attention_modules", True
                ),
                finetune_mlp_modules=lora_config.get(
                    "finetune_mlp_modules", True
                ),
            ),
        )
        return adapted_model

    def get_markers(self) -> Dict[str, str]:
        markers = self.config.get("markers", {})
        if not isinstance(markers, (dict, DictConfig)):
            logger.warning(
                f"Markers in config is not a dict or DictConfig, but "
                f"{type(markers)}. Returning empty dict."
            )
            return {}
        # Ensure all keys and values in markers are strings for Dict[str, str]
        if not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in markers.items()
        ):
            logger.warning(
                "Markers dictionary contains non-str keys/values. Filtering."
            )
            return {
                k: v
                for k, v in markers.items()
                if isinstance(k, str) and isinstance(v, str)
            }
        return cast(Dict[str, str], markers)

    def save_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        save_path: str,
        adapter_name: Optional[str] = None,
    ) -> None:
        logger.info(f"Saving model and tokenizer to {save_path}")
        if hasattr(model, "save_pretrained") and callable(
            model.save_pretrained
        ):
            if adapter_name and hasattr(model, "active_adapter"):
                logger.info(f"Saving adapter '{adapter_name}'")
                model.save_pretrained(save_path, safe_serialization=True)
            else:
                model.save_pretrained(save_path, safe_serialization=True)
        else:
            logger.error(
                "Model object does not have a 'save_pretrained' method."
            )

        if hasattr(tokenizer, "save_pretrained") and callable(
            tokenizer.save_pretrained
        ):
            tokenizer.save_pretrained(save_path)
        else:
            logger.error(
                "Tokenizer object does not have a 'save_pretrained' method."
            )
        logger.info(f"Model and tokenizer saved to {save_path}")
