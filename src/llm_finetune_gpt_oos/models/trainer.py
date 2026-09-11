import inspect
import json
import math
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from ..config import load_config
from .base_model import load_base_model, load_tokenizer, prepare_lora_model


class ConversationDataset(Dataset):
    def __init__(self, path: str, tokenizer: Any, max_seq_length: int, cache_dir: str | None = None):
        self.rows = load_dataset("json", data_files=path, split="train", cache_dir=cache_dir, keep_in_memory=False)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        messages = self.rows[index]["messages"]
        prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        completion = messages[-1]["content"] + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"][: self.max_seq_length]
        prompt_budget = self.max_seq_length - len(completion_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def build_training_arguments(training: dict[str, Any]) -> TrainingArguments:
    values: dict[str, Any] = {"output_dir": training["output_dir"], "per_device_train_batch_size": training["per_device_train_batch_size"], "per_device_eval_batch_size": training["per_device_eval_batch_size"], "gradient_accumulation_steps": training["gradient_accumulation_steps"], "learning_rate": training["learning_rate"], "num_train_epochs": training["num_train_epochs"], "warmup_ratio": training["warmup_ratio"], "lr_scheduler_type": training["lr_scheduler_type"], "weight_decay": training["weight_decay"], "logging_steps": training["logging_steps"], "save_steps": training["save_steps"], "save_total_limit": training["save_total_limit"], "bf16": training["bf16"], "fp16": training["fp16"], "gradient_checkpointing": training["gradient_checkpointing"], "report_to": training["report_to"], "remove_unused_columns": False, "logging_strategy": "steps", "save_strategy": "steps", "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False}
    eval_parameter = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    values[eval_parameter] = "steps"
    values["eval_steps"] = training["eval_steps"]
    if training.get("max_steps") is not None:
        values["max_steps"] = training["max_steps"]
    return TrainingArguments(**values)


class SalesQLoRATrainer:
    def __init__(self, config_path: str = "configs/train.yaml"):
        self.config = load_config(config_path)
        self.model_config = self.config["model"]
        self.training_config = self.config["training"]
        self.output_dir = Path(self.training_config["output_dir"])

    def train(self) -> dict[str, Any]:
        set_seed(self.training_config["seed"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = load_tokenizer(self.model_config)
        train_dataset = ConversationDataset(self.training_config["train_file"], tokenizer, self.training_config["max_seq_length"], self.model_config.get("cache_dir"))
        validation_dataset = ConversationDataset(self.training_config["validation_file"], tokenizer, self.training_config["max_seq_length"], self.model_config.get("cache_dir"))
        if not len(train_dataset) or not len(validation_dataset):
            raise ValueError("Both train and validation datasets must contain at least one example")
        model = load_base_model(self.model_config, training=True)
        model = prepare_lora_model(model, self.model_config, self.training_config["gradient_checkpointing"])
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainer = Trainer(model=model, args=build_training_arguments(self.training_config), train_dataset=train_dataset, eval_dataset=validation_dataset, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8))
        result = trainer.train(resume_from_checkpoint=self.training_config.get("resume_from_checkpoint"))
        metrics = trainer.evaluate()
        final_adapter = self.output_dir / "final_adapter"
        trainer.save_model(final_adapter)
        tokenizer.save_pretrained(final_adapter)
        summary = {"base_model_id": self.model_config["base_model_id"], "adapter_path": str(final_adapter), "train_examples": len(train_dataset), "validation_examples": len(validation_dataset), "trainable_parameters": trainable_parameters, "total_parameters": total_parameters, "train_loss": result.training_loss, "eval_loss": metrics.get("eval_loss"), "perplexity": math.exp(min(metrics["eval_loss"], 20)) if metrics.get("eval_loss") is not None else None, "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        (self.output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
