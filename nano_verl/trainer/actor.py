"""Actor-side construction helpers.

This module groups everything that belongs to the policy model side:
- tokenizer / processor
- model loading kwargs
- optional LoRA / QLoRA config
"""

from __future__ import annotations

from typing import Any

from nano_verl.trainer.config import GRPOExperimentConfig


def build_model_init_kwargs(config: GRPOExperimentConfig) -> dict[str, Any]:
    """Prepare kwargs forwarded to `from_pretrained()`."""

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}

    if config.bf16:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for bf16 training.") from exc
        model_kwargs["torch_dtype"] = torch.bfloat16

    if config.load_in_4bit:
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required to construct the 4-bit quantization config."
            ) from exc

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    return model_kwargs


def build_processing_class(model_name: str):
    """Load the tokenizer / processing class used by TRL."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for tokenizer loading. Install requirements-real.txt first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_peft_config(config: GRPOExperimentConfig):
    """Create a LoRA config when PEFT is enabled."""

    if not config.use_peft:
        return None

    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("peft is required for LoRA training. Install requirements-real.txt first.") from exc

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )

