# ⚡ Quick Start Guide

## 🎯 What is this?

A **configuration-driven pipeline** for training LLMs on reasoning tasks using **RLVR** (Reinforcement Learning from Verifiable Rewards). Train models by editing YAML files - no code changes needed.

## 🚀 5-Minute Setup

### 1. Install Dependencies

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 2. Run Default Training

```bash
# Train Gemma-1B on GSM8K with GRPO
poetry run python run_pipeline.py
```

That's it! The pipeline will:
- Load GSM8K dataset
- Initialize Gemma-1B with LoRA adapters
- Train using GRPO with verifiable math rewards
- Log to Weights & Biases
- Save trained model

## 🔧 Common Modifications

### Change Model

```bash
# Use different model
python run_pipeline.py \
  model.model_name_or_path="unsloth/llama-3-8b-it"
```

### Adjust Training

```bash
# Quick training run
python run_pipeline.py \
  train.max_steps=100 \
  data.max_train_samples=500

# Memory-efficient training  
python run_pipeline.py \
  model.load_in_4bit=true \
  train.per_device_train_batch_size=4
```

### Switch Dataset

```bash
# Train on financial reasoning
python run_pipeline.py data_component=finqa
```

### Tune Rewards

```bash
# Increase correctness reward
python run_pipeline.py \
  reward.reward_functions[0].params.correct_reward=5.0
```

## 📊 Monitor Training

### Weights & Biases
- Automatic logging enabled by default
- View at: https://wandb.ai/your-project

### Local Logs
```bash
# View training logs
tail -f outputs/latest/run.log
```

## 🎛️ Key Configuration Files

```
conf/
├── config.yaml                    # Main pipeline config
├── model_component/default.yaml   # Model & LoRA settings
├── training_loop_component/       # GRPO parameters
├── reward_component/              # Reward functions
└── prompts/math_reasoning.yaml    # Reasoning prompts
```

## 🔍 Understanding RLVR

**Traditional RLHF**: Human feedback → subjective rewards
**RLVR**: Verifiable outcomes → objective rewards

### Verifiable Rewards in this Pipeline:
- ✅ **Mathematical Correctness**: Answer matches ground truth
- ✅ **Format Compliance**: Follows reasoning structure
- ✅ **Numerical Accuracy**: Precise calculation verification

### Example Reward Flow:
1. Model generates: `<start_working_out>2+3=5<end_working_out><SOLUTION>5</SOLUTION>`
2. Answer extraction: `5`
3. Ground truth: `5`
4. Reward: `+3.0` (correct) + `+3.0` (format) = `+6.0`

## 🛠️ Troubleshooting

### Out of Memory
```bash
python run_pipeline.py \
  model.load_in_4bit=true \
  train.per_device_train_batch_size=2 \
  train.gradient_accumulation_steps=8
```

### Slow Training
```bash
# Reduce dataset for testing
python run_pipeline.py \
  data.max_train_samples=100 \
  train.max_steps=50
```

### Configuration Errors
```bash
# Validate config without training
python -c "
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path='conf', config_name='config', version_base=None)
def test(cfg): print(OmegaConf.to_yaml(cfg))
test()
"
```

## 📚 Next Steps

1. **Read [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** for detailed config options
2. **Check [ARCHITECTURE.md](ARCHITECTURE.md)** to understand the system design  
3. **See [COMPONENT_GUIDE.md](COMPONENT_GUIDE.md)** to add custom components

## 💡 Pro Tips

- Start with small `max_steps` and `max_train_samples` for testing
- Monitor reward distributions in W&B to tune reward weights
- Use `load_in_4bit=true` for large models on limited GPU memory
- Check `outputs/latest/` for logs and model checkpoints

---

**Ready to train reasoning models? Just run `poetry run python run_pipeline.py`** 🚀 