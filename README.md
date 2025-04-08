# 🎨 Python Code Style Template

A robust template for enforcing consistent code style and linting in Python projects.

## ✨ Overview

This template provides a standardized foundation for maintaining clean, consistent Python code across your projects. It includes pre-configured linting and code style enforcement tools to ensure your codebase remains maintainable and professional.

## 🚀 Features

- Pre-configured linting setup
- Code style enforcement
- Python best practices
- Poetry for dependency management
- MIT Licensed for maximum flexibility

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/code-style-template.git
```

2. Install Poetry (if you haven't already):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies using Poetry:
```bash
cd code-style-template
poetry install
```

## 🛠️ Usage

Use this template as a starting point for your Python projects to maintain consistent code quality:

1. Copy the configuration files to your project
2. Install dependencies with `poetry install`
3. Activate the virtual environment with `poetry shell`
4. Start coding with automatic style enforcement!

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch
3. Install dependencies with `poetry install`
4. Make your changes
5. Run tests and linting
6. Commit your changes
7. Push to the branch
8. Open a Pull Request

## ⭐ Support

If you find this template useful, please consider giving it a star!

# GSM8K GRPO Training

This repository contains modularized code for training language models on the GSM8K dataset using the GRPO (Generative Reward in Policy Optimization) approach.

## Features

- Fine-tuning LLMs on math reasoning tasks using GRPO
- Modular design following PEP8 standards
- Poetry for dependency management
- Support for various model architectures
- Custom reward functions targeting math reasoning
- Inference capabilities for testing trained models

## Installation

This project uses Poetry for dependency management:

```bash
# Install poetry if you don't have it already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

## Project Structure

- `src/`: Main package
  - `model.py`: Model initialization and LoRA configuration
  - `data.py`: Dataset loading and processing
  - `rewards.py`: Reward functions for GRPO
  - `config.py`: Training configuration and constants
  - `inference.py`: Inference utilities 
  - `train.py`: Main training logic

- `train_gsm8k.py`: Command-line script to run training
- `inference.py`: Command-line script to run inference

## Usage

### Training

To train a model with default parameters:

```bash
poetry run python train_gsm8k.py
```

With custom parameters:

```bash
poetry run python train_gsm8k.py \
  --model_name "unsloth/gemma-3-4b-it-unsloth-bnb-4bit" \
  --max_seq_length 2048 \
  --load_in_4bit \
  --output_dir "outputs/gemma-4b" \
  --save_model_path "gemma-4b" \
  --run_inference
```

### Inference

To run inference with a trained model:

```bash
poetry run python inference.py \
  --model_path "gemma-3" \
  --query "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4 eggs. How many eggs does she have left each day?" \
  --max_new_tokens 256
```

## Customization

You can customize various aspects of the training:

- Model architecture by changing the `model_name` parameter
- LoRA parameters in `model.py`
- Reward functions in `rewards.py` 
- Training configuration in `config.py`
- System prompt and reasoning markers in `config.py`

## Profiling and Optimization
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --output=nv_report.qdrep \
  --force-overwrite=true \
  --stats=true \
  python main.py

## License

See the LICENSE file for more information.
