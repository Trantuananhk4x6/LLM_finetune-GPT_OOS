from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def torch_dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def load_tokenizer(model_config: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model_id"], cache_dir=model_config.get("cache_dir"), trust_remote_code=model_config.get("trust_remote_code", False))
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_config: dict[str, Any], training: bool) -> AutoModelForCausalLM:
    quantization = model_config.get("quantization", {})
    kwargs: dict[str, Any] = {"cache_dir": model_config.get("cache_dir"), "trust_remote_code": model_config.get("trust_remote_code", False), "low_cpu_mem_usage": True}
    if quantization.get("load_in_4bit", False):
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit loading requires a CUDA-enabled PyTorch installation")
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=quantization.get("quant_type", "nf4"), bnb_4bit_compute_dtype=torch_dtype(quantization.get("compute_dtype", "bfloat16")), bnb_4bit_use_double_quant=quantization.get("double_quant", True))
        if not training:
            kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if model_config.get("use_flash_attention", False):
        kwargs["attn_implementation"] = "flash_attention_2"
    return AutoModelForCausalLM.from_pretrained(model_config["base_model_id"], **kwargs)


def prepare_lora_model(model: AutoModelForCausalLM, model_config: dict[str, Any], gradient_checkpointing: bool) -> AutoModelForCausalLM:
    lora = model_config["lora"]
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    model.config.use_cache = False
    return get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM"))


def load_adapter(model_config: dict[str, Any], adapter_path: str) -> tuple[Any, Any]:
    tokenizer = load_tokenizer(model_config)
    model = load_base_model(model_config, training=False)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer
