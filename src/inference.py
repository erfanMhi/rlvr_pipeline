"""Inference utilities for trained models."""

from typing import Any, Union

from transformers import TextStreamer


def format_chat_prompt(
    tokenizer: Any, system_prompt: str, user_prompt: str  # type: ignore
) -> Union[Any, str]:
    """
    Format a chat prompt for model inference.

    Args:
        tokenizer: The tokenizer
        system_prompt: The system prompt
        user_prompt: The user prompt/question

    Returns:
        Formatted chat template text
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def generate_response(
    model: Any,
    tokenizer: Any,
    formatted_prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    stream: bool = True,
) -> Union[str, Any]:
    """
    Generate a response from the model.

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        formatted_prompt: The formatted prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        stream: Whether to stream the output

    Returns:
        The generated text if stream=False, otherwise None
    """
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        streamer=streamer,
    )

    if not stream:
        output_tokens = output[0][inputs.input_ids.shape[1] :]
        return tokenizer.decode(output_tokens, skip_special_tokens=True)
    return None
