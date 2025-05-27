# ⚙️ Configuration Guide

## Overview

This guide explains how to configure the RL training pipeline for **Reinforcement Learning from Verifiable Rewards (RLVR)** - specifically designed for reasoning tasks where rewards can be computed from verifiable outcomes (e.g., mathematical correctness).

## 🎯 Core Configuration Concepts

### Hydra Configuration Structure

```yaml
defaults:
  - data_component@data: default           # Dataset loading
  - model_component@model: default         # Model & adapters  
  - training_loop_component@train: default # GRPO training
  - reward_component@reward: default       # Verifiable rewards
  - evaluation_component@eval: default     # Evaluation metrics
  - prompts@prompts: math_reasoning        # Prompt templates
```

### Configuration Override Patterns

```bash
# Override specific parameters
python run_pipeline.py model.max_seq_length=2048

# Switch component configurations
python run_pipeline.py data_component=custom_dataset

# Multiple overrides
python run_pipeline.py \
  model.lora_config.r=16 \
  train.learning_rate=1e-5 \
  train.max_steps=2000
```

## 📊 Component Configurations

### 1. Data Component

**Purpose**: Load and format datasets for reasoning tasks

```yaml
# conf/data_component/gsm8k.yaml
_target_: src.components.data.default_data_component.DefaultDataComponent
dataset_name: "gsm8k"
dataset_path: "openai/gsm8k"
dataset_config_name: "main"
max_train_samples: null  # Use full dataset
max_eval_samples: 500    # Limit eval for speed
```

**Key Parameters**:
- `dataset_name`: Processor type (`gsm8k`, `finqa`)
- `dataset_path`: HuggingFace dataset path
- `max_train_samples`: Limit training data (null = all)

### 2. Model Component

**Purpose**: Configure model, tokenizer, and adapters

```yaml
# conf/model_component/gemma_1b_lora.yaml
_target_: src.components.model.default_model_component.DefaultModelComponent
model_name_or_path: "unsloth/gemma-3-1b-it"
max_seq_length: 1536
load_in_4bit: false

# LoRA configuration
lora_config:
  use_lora: true
  r: 8                    # Rank (higher = more parameters)
  lora_alpha: 16          # Scaling factor (typically 2*r)
  lora_dropout: 0.0
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]

# Model-specific markers for reasoning
markers: ${get_model_markers_dict:}
save_model_path: "./output/trained_model"
```

**Key Parameters**:
- `model_name_or_path`: HuggingFace model identifier
- `max_seq_length`: Maximum context length
- `lora_config.r`: LoRA rank (8-64 typical range)
- `markers`: Model-specific formatting tokens

### 3. Training Loop Component (GRPO)

**Purpose**: Configure GRPO training algorithm

```yaml
# conf/training_loop_component/grpo_default.yaml
_target_: src.components.training_loop.default_training_loop.DefaultTrainingLoopComponent
output_dir: "./output/checkpoints"
learning_rate: 5e-6
max_steps: 1000
max_prompt_length: 512
max_completion_length: 1024  # Computed: max_seq_length - max_prompt_length

# Batch configuration
per_device_train_batch_size: 16
gradient_accumulation_steps: 1

# Optimization
warmup_ratio: 0.1
lr_scheduler_type: "cosine"
optim_name: "adamw_torch_fused"

# GRPO-specific
num_generations: 16          # Samples per prompt for GRPO
log_completions_to_wandb: true
```

**Key Parameters**:
- `learning_rate`: Typically 1e-6 to 1e-5 for RL
- `num_generations`: GRPO samples per prompt
- `max_completion_length`: Response length limit

### 4. Reward Component (RLVR)

**Purpose**: Define verifiable reward functions for reasoning

```yaml
# conf/reward_component/tasks/math_reasoning.yaml
reward_functions:
  # Correctness rewards (verifiable)
  - category: "answer_matching"
    type: "gsm8k_answer_check"
    params:
      correct_reward: 3.0
      numerical_close_reward_strong: 0.5
      wrong_penalty: -1.0
  
  # Format rewards (structured reasoning)
  - category: "format_checking"
    type: "match_format_exactly"
    params:
      reward_value: 3.0
```

**Reward Categories**:
- **Answer Matching**: Verifiable correctness rewards
  - `gsm8k_answer_check`: Numerical answer verification
  - `finqa_numerical_match`: Financial calculation verification
  - `check_numbers`: Basic numerical extraction
- **Format Checking**: Structured reasoning rewards
  - `match_format_exactly`: Strict format compliance
  - `match_format_approximately`: Flexible format checking

### 5. Evaluation Component

**Purpose**: Monitor training progress and final performance

```yaml
# conf/evaluation_component/default.yaml
_target_: src.components.evaluation.default_evaluation_component.DefaultEvaluationComponent
enabled: true
eval_datasets_names: ["gsm8k"]
eval_steps: 100              # Evaluate every N steps
eval_max_new_tokens: 256
eval_num_samples: 200        # Limit eval dataset size
eval_batch_size: 8
```

### 6. Prompts Configuration

**Purpose**: Define reasoning prompt templates

```yaml
# conf/prompts/math_reasoning.yaml
prompting_mode: reasoning_mode

system_prompts:
  reasoning_mode: >
    You are given a problem. Think about the problem and provide your working out.
    Place it between ${get_marker:reasoning_start} and ${get_marker:reasoning_end}.
    Then, provide your solution between ${get_marker:solution_start} 
    and ${get_marker:solution_end}.
```

## 🚀 Common Configuration Patterns

### Quick Start: GSM8K Training

```bash
# Basic GRPO training on GSM8K
python run_pipeline.py \
  data_component=gsm8k \
  model_component=gemma_1b_lora \
  train.max_steps=500
```

### Memory-Efficient Training

```bash
# 4-bit quantization + smaller batch
python run_pipeline.py \
  model.load_in_4bit=true \
  train.per_device_train_batch_size=8 \
  train.gradient_accumulation_steps=2
```

### Custom Reward Weights

```bash
# Adjust reward balance
python run_pipeline.py \
  reward.reward_functions[0].params.correct_reward=5.0 \
  reward.reward_functions[1].params.reward_value=2.0
```

### Distributed Training

```bash
# Multi-GPU training
torchrun --nproc_per_node=4 run_pipeline.py \
  train.per_device_train_batch_size=4
```

## 🔧 Advanced Configuration

### Custom Dataset Integration

1. **Create dataset processor**:
```python
class CustomDatasetProcessor(AbstractDatasetProcessor):
    def load_raw_dataset(self, split: str) -> Dataset:
        return load_dataset("your/dataset", split=split)
    
    def get_question_text(self, example: Dict[str, Any]) -> str:
        return example["question"]
    
    def extract_ground_truth_answer_from_example(self, example: Dict[str, Any]) -> Optional[str]:
        return example["answer"]
```

2. **Register in data component**:
```python
DATASET_PROCESSORS = {
    "custom_dataset": CustomDatasetProcessor,
    # ... existing processors
}
```

3. **Create configuration**:
```yaml
# conf/data_component/custom.yaml
_target_: src.components.data.default_data_component.DefaultDataComponent
dataset_name: "custom_dataset"
dataset_path: "your/dataset"
```

### Custom Reward Functions

1. **Define reward function**:
```python
def custom_reward(prompts, completions, answers, **kwargs):
    scores = []
    for completion, answer in zip(completions, answers):
        # Your verification logic
        score = verify_custom_logic(completion, answer)
        scores.append(score)
    return scores
```

2. **Register in reward component**:
```python
def _build_custom_reward_fn(self, fn_config, markers):
    if fn_config.get("type") == "custom_verification":
        return partial(custom_reward, **fn_config.get("params", {}))
```

### Environment-Specific Configs

```yaml
# conf/config_dev.yaml - Development
defaults:
  - config
  - override data: small_dataset
  - override train: quick_train

# conf/config_prod.yaml - Production  
defaults:
  - config
  - override train: full_train
  - override eval: comprehensive_eval
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**:
```bash
# Reduce batch size and use gradient accumulation
python run_pipeline.py \
  train.per_device_train_batch_size=4 \
  train.gradient_accumulation_steps=4 \
  model.load_in_4bit=true
```

2. **Slow Training**:
```bash
# Reduce dataset size for testing
python run_pipeline.py \
  data.max_train_samples=1000 \
  train.max_steps=100
```

3. **Reward Function Errors**:
```bash
# Check marker configuration
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('conf/model_component/default.yaml')
print('Markers:', cfg.markers)
"
```

### Configuration Validation

```bash
# Test configuration without training
python run_pipeline.py \
  --config-name=config \
  --dry-run
```

### Debug Mode

```bash
# Enable verbose logging
python run_pipeline.py \
  logging_level=DEBUG \
  train.logging_steps=1
```

## 📈 Performance Tuning

### Memory Optimization
- Use `load_in_4bit=true` for large models
- Reduce `max_seq_length` if possible
- Lower `per_device_train_batch_size`

### Speed Optimization  
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use `optim_name="adamw_torch_fused"` for faster optimization
- Enable `bf16=true` on supported hardware

### Reward Tuning
- Balance positive/negative rewards (typically 3:1 ratio)
- Start with simple rewards, add complexity gradually
- Monitor reward distributions in W&B logs

This configuration system enables rapid experimentation with RLVR algorithms while maintaining reproducibility and scalability. 