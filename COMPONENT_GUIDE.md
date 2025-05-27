# 🔧 Component Development Guide

## Overview

This guide provides detailed instructions for developing new components in the RL Training Pipeline. The component system is designed to be highly extensible, allowing developers to add new functionality without modifying the core orchestration logic.

## 🏗️ Component Architecture Patterns

### Base Component Interface

All components inherit from the `BaseComponent` abstract class:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseComponent(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the component-specific configuration."""
        pass
```

### Component Lifecycle

1. **Instantiation**: Component is created with configuration dictionary
2. **Validation**: `validate_config()` is called to ensure configuration is valid
3. **Initialization**: Component-specific setup is performed
4. **Execution**: Component methods are called during pipeline execution
5. **Cleanup**: Resources are released (if applicable)

## 📦 Component Types

### 1. Data Components

**Purpose**: Handle dataset loading, preprocessing, and formatting

**Interface**: `DataComponentInterface`

```python
from abc import abstractmethod
from typing import Optional
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from src.components.base_component import BaseComponent

class DataComponentInterface(BaseComponent):
    @abstractmethod
    def load_and_prepare_data(
        self,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str,
        split: str = "train",
        dataset_name_override: Optional[str] = None,
    ) -> Dataset:
        """Loads, processes, and formats the dataset."""
        pass
```

**Implementation Example**:

```python
class CustomDataComponent(DataComponentInterface):
    def validate_config(self) -> bool:
        required_keys = ["dataset_name", "dataset_path"]
        return all(key in self.config for key in required_keys)
    
    def load_and_prepare_data(
        self,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str,
        split: str = "train",
        dataset_name_override: Optional[str] = None,
    ) -> Dataset:
        # Implementation logic
        dataset_name = dataset_name_override or self.config["dataset_name"]
        # Load and process dataset
        return processed_dataset
```

**Configuration Template**:

```yaml
# conf/data_component/custom_data.yaml
_target_: src.components.data.custom_data_component.CustomDataComponent
dataset_name: "custom_dataset"
dataset_path: "path/to/dataset"
preprocessing_config:
  max_length: 512
  truncation: true
custom_parameters:
  specific_param: value
```

### 2. Model Components

**Purpose**: Manage model initialization, adapters, and serialization

**Interface**: `ModelComponentInterface`

```python
from abc import abstractmethod
from typing import Dict, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from src.components.base_component import BaseComponent

class ModelComponentInterface(BaseComponent):
    @abstractmethod
    def initialize_model_and_tokenizer(
        self,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Initializes and returns the model and tokenizer."""
        pass

    @abstractmethod
    def add_adapters(self, model: PreTrainedModel) -> PreTrainedModel:
        """Adds PEFT adapters (e.g., LoRA) to the model if configured."""
        pass

    @abstractmethod
    def get_markers(self) -> Dict[str, str]:
        """Returns model-specific markers for prompt formatting."""
        pass

    @abstractmethod
    def save_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        save_path: str,
        adapter_name: Optional[str] = None,
    ) -> None:
        """Saves the model and tokenizer to the specified path."""
        pass
```

**Implementation Example**:

```python
class CustomModelComponent(ModelComponentInterface):
    def validate_config(self) -> bool:
        return "model_name_or_path" in self.config
    
    def initialize_model_and_tokenizer(
        self,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        model_name = self.config["model_name_or_path"]
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Apply adapters if configured
        if self.config.get("use_adapters", False):
            model = self.add_adapters(model)
        
        return model, tokenizer
    
    def add_adapters(self, model: PreTrainedModel) -> PreTrainedModel:
        # Implement adapter logic (LoRA, etc.)
        return model
    
    def get_markers(self) -> Dict[str, str]:
        return self.config.get("markers", {})
    
    def save_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        save_path: str,
        adapter_name: Optional[str] = None,
    ) -> None:
        # Implement model saving logic
        pass
```

### 3. Training Loop Components

**Purpose**: Implement various RL training algorithms

**Interface**: `TrainingLoopInterface`

```python
from abc import abstractmethod
from typing import Callable, List, Optional
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from src.components.base_component import BaseComponent

class TrainingLoopInterface(BaseComponent):
    @abstractmethod
    def train(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        reward_functions: Optional[List[Callable]],
        callbacks: Optional[List[TrainerCallback]],
    ) -> None:
        """Executes the training loop using configuration from self.config."""
        pass
```

**Implementation Example**:

```python
class CustomRLTrainingLoop(TrainingLoopInterface):
    def validate_config(self) -> bool:
        required_keys = ["learning_rate", "max_steps"]
        return all(key in self.config for key in required_keys)
    
    def train(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        reward_functions: Optional[List[Callable]],
        callbacks: Optional[List[TrainerCallback]],
    ) -> None:
        # Implement custom RL training algorithm
        learning_rate = self.config["learning_rate"]
        max_steps = self.config["max_steps"]
        
        # Setup optimizer, scheduler, etc.
        optimizer = self._setup_optimizer(model, learning_rate)
        scheduler = self._setup_scheduler(optimizer)
        
        # Training loop implementation
        for step in range(max_steps):
            # Custom training logic
            pass
    
    def _setup_optimizer(self, model, learning_rate):
        # Optimizer setup logic
        pass
    
    def _setup_scheduler(self, optimizer):
        # Scheduler setup logic
        pass
```

### 4. Reward Components

**Purpose**: Define reward functions for RL training

**Interface**: `RewardComponentInterface`

```python
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional
from src.components.base_component import BaseComponent

class RewardComponentInterface(BaseComponent):
    @abstractmethod
    def get_reward_pipelines(
        self,
        model_info: Optional[Dict[str, Any]],
        reward_functions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Callable[..., Any]]:
        """Constructs and returns a list of reward functions."""
        pass
```

**Implementation Example**:

```python
class CustomRewardComponent(RewardComponentInterface):
    def validate_config(self) -> bool:
        return "reward_functions" in self.config
    
    def get_reward_pipelines(
        self,
        model_info: Optional[Dict[str, Any]],
        reward_functions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Callable[..., Any]]:
        reward_configs = reward_functions or self.config["reward_functions"]
        pipelines = []
        
        for reward_config in reward_configs:
            reward_type = reward_config["type"]
            if reward_type == "custom_reward":
                pipeline = self._create_custom_reward(reward_config, model_info)
                pipelines.append(pipeline)
        
        return pipelines
    
    def _create_custom_reward(self, config, model_info):
        def reward_function(prompt, response, ground_truth):
            # Implement custom reward logic
            score = self._calculate_reward(response, ground_truth)
            return score
        
        return reward_function
    
    def _calculate_reward(self, response, ground_truth):
        # Custom reward calculation
        return 1.0  # Placeholder
```

### 5. Evaluation Components

**Purpose**: Handle model evaluation during and after training

**Interface**: `EvaluationComponentInterface`

```python
from abc import abstractmethod
from typing import Optional
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerState,
)
from src.components.base_component import BaseComponent

class EvaluationComponentInterface(BaseComponent):
    @abstractmethod
    def get_trainer_callback(
        self,
        data_component_instance,
        model_component_instance,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Optional[TrainerCallback]:
        """Returns a TrainerCallback for in-training evaluation."""
        pass

    @abstractmethod
    def on_evaluation_run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        trainer_state: TrainerState,
    ) -> None:
        """Called by orchestrator for post-training evaluation."""
        pass
```

### 6. Observer Components

**Purpose**: Provide monitoring, logging, and experiment tracking

**Interface**: `ObserverInterface`

```python
from abc import abstractmethod
from typing import Any, Dict, Optional
from src.components.base_component import BaseComponent

class ObserverInterface(BaseComponent):
    @abstractmethod
    def on_pipeline_start(self, orchestrator_config: Dict[str, Any]) -> None:
        """Called when the main pipeline starts."""
        pass

    @abstractmethod
    def on_pipeline_end(
        self, status: str, error: Optional[Exception] = None
    ) -> None:
        """Called when the main pipeline ends."""
        pass

    @abstractmethod
    def on_step_start(
        self, step_name: str, step_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called when a specific pipeline stage starts."""
        pass

    @abstractmethod
    def on_step_end(
        self,
        step_name: str,
        output: Optional[Any] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Called when a specific pipeline stage ends."""
        pass
```

## 🔧 Development Workflow

### Step 1: Define Component Interface

If creating a new component type, first define the interface:

```python
# src/components/new_component/interface.py
from abc import abstractmethod
from src.components.base_component import BaseComponent

class NewComponentInterface(BaseComponent):
    @abstractmethod
    def new_component_method(self) -> Any:
        """Component-specific method."""
        pass
```

### Step 2: Implement Component

Create the concrete implementation:

```python
# src/components/new_component/default_new_component.py
import logging
from typing import Any, Dict

from src.components.new_component.interface import NewComponentInterface

logger = logging.getLogger(__name__)

class DefaultNewComponent(NewComponentInterface):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.component_state = None
    
    def validate_config(self) -> bool:
        """Validate component configuration."""
        required_keys = ["required_param"]
        missing_keys = [key for key in required_keys if key not in self.config]
        
        if missing_keys:
            logger.error(f"Missing required config keys: {missing_keys}")
            return False
        
        # Additional validation logic
        return True
    
    def new_component_method(self) -> Any:
        """Implement component-specific functionality."""
        param_value = self.config["required_param"]
        # Implementation logic
        return result
```

### Step 3: Create Configuration Schema

Define the configuration structure:

```yaml
# conf/new_component/default.yaml
_target_: src.components.new_component.default_new_component.DefaultNewComponent
required_param: "default_value"
optional_param: 42
nested_config:
  sub_param1: true
  sub_param2: "nested_value"
```

### Step 4: Update Orchestrator (if needed)

For new component types, update the orchestrator:

```python
# src/orchestration/pipeline_orchestrator.py
def _init_new_component(self) -> None:
    if "new_component" not in self.pipeline_config:
        logger.info("New component not configured, skipping.")
        return
    
    config = self.pipeline_config["new_component"]
    self.new_component = DefaultNewComponent(config)
    
    if not self.new_component.validate_config():
        raise ValueError("New component configuration is invalid.")
    
    logger.info("New component initialized.")

def run(self) -> None:
    # Add to pipeline execution flow
    self._init_new_component()
    # ... rest of pipeline
```

### Step 5: Add Tests

Create comprehensive tests:

```python
# tests/components/new_component/test_default_new_component.py
import pytest
from src.components.new_component.default_new_component import DefaultNewComponent

class TestDefaultNewComponent:
    def test_validate_config_success(self):
        config = {"required_param": "test_value"}
        component = DefaultNewComponent(config)
        assert component.validate_config()
    
    def test_validate_config_missing_required(self):
        config = {}
        component = DefaultNewComponent(config)
        assert not component.validate_config()
    
    def test_new_component_method(self):
        config = {"required_param": "test_value"}
        component = DefaultNewComponent(config)
        result = component.new_component_method()
        assert result is not None
```

## 🎯 Best Practices

### Configuration Design

1. **Use Clear Naming**: Configuration keys should be descriptive and consistent
2. **Provide Defaults**: Include sensible default values where possible
3. **Validate Early**: Implement thorough configuration validation
4. **Document Parameters**: Include comments explaining configuration options

```yaml
# Good configuration example
model_component:
  # Model identifier from HuggingFace Hub
  model_name_or_path: "unsloth/gemma-3-1b-it"
  
  # Maximum sequence length for training
  max_seq_length: 1536
  
  # Quantization settings
  load_in_4bit: false  # Enable 4-bit quantization for memory efficiency
  load_in_8bit: false  # Enable 8-bit quantization (alternative to 4-bit)
  
  # LoRA configuration
  lora_config:
    use_lora: true      # Enable LoRA fine-tuning
    r: 8                # LoRA rank (higher = more parameters)
    lora_alpha: 16      # LoRA scaling factor (typically 2*r)
    lora_dropout: 0.0   # Dropout for LoRA layers
```

### Error Handling

1. **Graceful Degradation**: Handle errors gracefully without crashing the pipeline
2. **Informative Messages**: Provide clear error messages with context
3. **Logging**: Use appropriate logging levels for different scenarios

```python
def validate_config(self) -> bool:
    try:
        # Validation logic
        if "required_param" not in self.config:
            logger.error(
                f"Missing required parameter 'required_param' in "
                f"{self.__class__.__name__} configuration"
            )
            return False
        
        # Type validation
        if not isinstance(self.config["required_param"], str):
            logger.error(
                f"Parameter 'required_param' must be a string, "
                f"got {type(self.config['required_param'])}"
            )
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}", exc_info=True)
        return False
```

### Performance Considerations

1. **Lazy Loading**: Load resources only when needed
2. **Memory Management**: Clean up resources properly
3. **Caching**: Cache expensive computations when appropriate

```python
class OptimizedComponent(ComponentInterface):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._cached_resource = None
    
    @property
    def expensive_resource(self):
        """Lazy loading of expensive resource."""
        if self._cached_resource is None:
            self._cached_resource = self._load_expensive_resource()
        return self._cached_resource
    
    def _load_expensive_resource(self):
        # Expensive loading logic
        pass
    
    def cleanup(self):
        """Clean up resources."""
        if self._cached_resource is not None:
            # Cleanup logic
            self._cached_resource = None
```

### Testing Strategies

1. **Unit Tests**: Test individual component methods
2. **Integration Tests**: Test component interaction with others
3. **Configuration Tests**: Test various configuration scenarios
4. **Mock Dependencies**: Use mocks for external dependencies

```python
import pytest
from unittest.mock import Mock, patch

class TestComponentIntegration:
    @patch('src.components.external_dependency.ExternalService')
    def test_component_with_external_service(self, mock_service):
        # Setup mock
        mock_service.return_value.method.return_value = "mocked_result"
        
        # Test component
        config = {"service_config": "test"}
        component = MyComponent(config)
        result = component.method_using_external_service()
        
        # Assertions
        assert result == "expected_result"
        mock_service.assert_called_once()
```

## 🔍 Debugging and Troubleshooting

### Common Issues

1. **Configuration Errors**: Missing or invalid configuration parameters
2. **Import Errors**: Incorrect module paths in `_target_` specifications
3. **Type Mismatches**: Incorrect data types in configuration
4. **Resource Conflicts**: Multiple components trying to use same resources

### Debugging Tools

1. **Verbose Logging**: Enable detailed logging for troubleshooting
2. **Configuration Inspection**: Print resolved configurations
3. **Component State**: Log component state at key points

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# In component implementation
logger.debug(f"Component initialized with config: {self.config}")
logger.debug(f"Component state: {self.component_state}")
```

### Testing Configuration

```bash
# Test configuration resolution
python -c "
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='conf', config_name='config', version_base=None)
def test_config(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

test_config()
"
```

## 📚 Advanced Topics

### Custom Hydra Resolvers

Create custom resolvers for dynamic configuration:

```python
# src/utils/custom_hydra_resolvers.py
from omegaconf import OmegaConf

def register_custom_resolvers():
    OmegaConf.register_new_resolver(
        "get_model_config",
        lambda model_name: get_model_specific_config(model_name)
    )
    
    OmegaConf.register_new_resolver(
        "compute_derived_value",
        lambda base_value, multiplier: base_value * multiplier
    )

def get_model_specific_config(model_name: str) -> dict:
    # Logic to retrieve model-specific configuration
    return {"param": "value"}
```

### Component Dependencies

Handle dependencies between components:

```python
class DependentComponent(ComponentInterface):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dependencies = {}
    
    def set_dependency(self, name: str, component: BaseComponent):
        """Set a dependency component."""
        self.dependencies[name] = component
    
    def get_dependency(self, name: str) -> BaseComponent:
        """Get a dependency component."""
        if name not in self.dependencies:
            raise ValueError(f"Dependency '{name}' not found")
        return self.dependencies[name]
```

### Plugin System

Implement a plugin system for dynamic component loading:

```python
# src/utils/plugin_loader.py
import importlib
from typing import Type, Dict, Any

class PluginLoader:
    @staticmethod
    def load_component(target_path: str, config: Dict[str, Any]) -> BaseComponent:
        """Dynamically load component from target path."""
        module_path, class_name = target_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
        return component_class(config)
```

This comprehensive guide provides the foundation for developing robust, extensible components within the RL training pipeline framework. 