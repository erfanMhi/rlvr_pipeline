"""Model initialization and configuration."""

from typing import Any, Optional, Tuple

from unsloth import FastModel


def initialize_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    full_finetuning: bool = False,
) -> Tuple[Any, Any]:
    """
    Initialize and load the model from pretrained.

    Args:
        model_name: Name of the model to load
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load in 4bit quantization
        load_in_8bit: Whether to load in 8bit quantization
        full_finetuning: Whether to do full finetuning

    Returns:
        Tuple of (model, tokenizer)
    """
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        full_finetuning=full_finetuning,
        attn_implementation="sdpa",
    )
    return model, tokenizer


def add_lora_adapters(
    model: Any,
    finetune_vision_layers: bool = False,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0,
    bias: str = "none",
    random_state: int = 3407,
) -> Any:
    """
    Add LoRA adapters to the model.

    Args:
        model: The model to add LoRA adapters to
        finetune_vision_layers: Whether to finetune vision layers
        finetune_language_layers: Whether to finetune language layers
        finetune_attention_modules: Whether to finetune attention modules
        finetune_mlp_modules: Whether to finetune MLP modules
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        bias: Bias type
        random_state: Random state

    Returns:
        Model with LoRA adapters
    """
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=finetune_language_layers,
        finetune_attention_modules=finetune_attention_modules,
        finetune_mlp_modules=finetune_mlp_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        random_state=random_state,
    )
    return model


def save_model(
    model: Any,
    tokenizer: Any,
    local_path: str,
    save_full: bool = False,
    save_gguf: bool = False,
    quantization_type: str = "Q8_0",
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    """
    Save the model locally or push to hub.

    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        local_path: Local path to save the model
        save_full: Whether to save the full model (not just LoRA adapters)
        save_gguf: Whether to save in GGUF format
        quantization_type: Quantization type for GGUF
        push_to_hub: Whether to push to hub
        repo_id: Repository ID for pushing to hub
        token: Hugging Face token
    """
    if save_full:
        model.save_pretrained_merged(local_path, tokenizer)
        if push_to_hub and repo_id:
            model.push_to_hub_merged(repo_id, tokenizer, token=token)
    elif save_gguf:
        model.save_pretrained_gguf(
            local_path, quantization_type=quantization_type
        )
        if push_to_hub and repo_id:
            model.push_to_hub_gguf(
                local_path,
                quantization_type=quantization_type,
                repo_id=repo_id,
                token=token,
            )
    else:
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        if push_to_hub and repo_id:
            model.push_to_hub(repo_id, token=token)
            tokenizer.push_to_hub(repo_id, token=token)
