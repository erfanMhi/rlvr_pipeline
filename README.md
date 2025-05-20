# 🧠 RLVR Training Pipeline

A **composable component orchestrator** for **Reinforcement Learning from Verifiable Rewards (RLVR)** training of Large Language Models on reasoning tasks.

## 🎯 Key Features

- **RLVR Specialization**: Built for reasoning tasks with verifiable outcomes
- **Zero-Code Configuration**: Train models by modifying YAML configs only  
- **GRPO Implementation**: Generative Reward in Policy Optimization
- **Verifiable Rewards**: Mathematical correctness, format compliance
- **Modular Architecture**: Swappable components for rapid experimentation

## 📦 Installation

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

## 🚀 Quick Start

### Basic Training

```bash
# Train on GSM8K with default GRPO configuration
poetry run python run_pipeline.py

# Custom configuration
poetry run python run_pipeline.py \
  model.model_name_or_path="unsloth/gemma-3-1b-it" \
  train.max_steps=500 \
  train.learning_rate=1e-5
```

### Configuration-Only Experiments

```bash
# Switch datasets
python run_pipeline.py data_component=finqa

# Adjust reward weights  
python run_pipeline.py \
  reward.reward_functions[0].params.correct_reward=5.0

# Memory-efficient training
python run_pipeline.py \
  model.load_in_4bit=true \
  train.per_device_train_batch_size=8
```

## 🏗️ Architecture

The pipeline consists of 6 composable components:

1. **Data Component**: Dataset loading and preprocessing (GSM8K, FinQA)
2. **Model Component**: Model initialization, LoRA adapters, quantization
3. **Training Loop**: GRPO algorithm implementation
4. **Reward Component**: Verifiable reward functions for reasoning
5. **Evaluation Component**: In-training and post-training evaluation
6. **Observer Component**: Experiment tracking (W&B integration)

## 📊 Supported Tasks

### Mathematical Reasoning
- **GSM8K**: Grade school math word problems
- **FinQA**: Financial reasoning and calculation

### Reward Types
- **Answer Matching**: Numerical correctness verification
- **Format Checking**: Structured reasoning compliance
- **Custom Rewards**: Extensible reward function system

## 🔧 Configuration

All behavior is controlled through Hydra YAML configurations:

```yaml
# conf/config.yaml
defaults:
  - data_component@data: default
  - model_component@model: default  
  - training_loop_component@train: default
  - reward_component@reward: default
  - evaluation_component@eval: default
  - prompts@prompts: math_reasoning
```

See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for detailed configuration options.

## 📚 Documentation

- [**ARCHITECTURE.md**](ARCHITECTURE.md): System design and component details
- [**COMPONENT_GUIDE.md**](COMPONENT_GUIDE.md): Component development guide
- [**CONFIGURATION_GUIDE.md**](CONFIGURATION_GUIDE.md): Configuration reference

## 🔬 Research Focus

This pipeline is optimized for **RLVR** (Reinforcement Learning from Verifiable Rewards) rather than traditional RLHF. Key differences:

- **Verifiable Outcomes**: Rewards computed from objective correctness
- **Reasoning Tasks**: Structured mathematical and logical reasoning
- **Format Compliance**: Reward structured thinking patterns
- **Deterministic Evaluation**: Reproducible reward calculation



## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Follow component development patterns in [COMPONENT_GUIDE.md](COMPONENT_GUIDE.md)
4. Add tests for new components
5. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built for reasoning. Configured for research. Optimized for results.**
