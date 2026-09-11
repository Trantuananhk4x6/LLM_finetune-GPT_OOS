"""
Kịch bản Fine-tuning Qwen 9B Smart Sales cho Google Colab, Kaggle, RunPod, Vast.ai (Linux/CUDA)
Hỗ trợ:
- Tự động nhận diện GPU (T4, L4, A100, V100, RTX 3090/4090)
- Tối ưu 4-bit QLoRA (BitsAndBytes NF4 + Gradient Checkpointing)
- Huấn luyện siêu nhanh, tiết kiệm VRAM (< 10GB VRAM, chạy tốt trên Colab T4 miễn phí)
- Tự động lưu checkpoint và xuất LoRA Adapter
"""
import argparse
import math
import os
import sys
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Cloud/Colab Trainer for Qwen Smart Sales")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                        help="Base Model ID trên Hugging Face (ví dụ: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3.5-9B, v.v.)")
    parser.add_argument("--train_file", type=str, default="data/processed/smart_sales_train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/processed/smart_sales_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="./artifacts/runs/qwen-smart-sales-cloud")
    parser.add_argument("--max_seq_length", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    return parser.parse_args()


def format_dataset(dataset_path: str, tokenizer, max_seq_length: int):
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    def process_row(example):
        messages = example["messages"]
        # Format conversation theo chuẩn chat template của Qwen
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        completion = messages[-1]["content"] + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"][:max_seq_length]

        prompt_budget = max_seq_length - len(completion_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

    return raw_dataset.map(process_row, remove_columns=raw_dataset.column_names)


def main():
    args = parse_args()
    set_seed(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print("  QWEN SMART SALES CLOUD TRAINER (COLAB / RUNPOD / KAGGLE)")
    print("==========================================================")
    print(f"[*] Base Model      : {args.model_id}")
    print(f"[*] Train Dataset   : {args.train_file}")
    print(f"[*] Output Dir      : {args.output_dir}")
    print(f"[*] GPU Device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Khuyên dùng GPU)'}")
    print("==========================================================")

    # 1. Cấu hình lượng tử hóa 4-bit BitsAndBytes (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Tải Tokenizer và Base Model
    print("[*] Đang nạp Tokenizer & Model 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. Cấu hình LoRA Adapter cho Qwen
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Chuẩn bị dữ liệu
    print("[*] Đang xử lý Tokenization cho dữ liệu...")
    train_dataset = format_dataset(args.train_file, tokenizer, args.max_seq_length)
    val_dataset = format_dataset(args.val_file, tokenizer, args.max_seq_length)
    print(f"[+] Số mẫu huấn luyện: {len(train_dataset)} | Số mẫu kiểm thử: {len(val_dataset)}")

    # 5. Cấu hình TrainingArguments tối ưu Cloud
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    print("[*] Bắt đầu quá trình huấn luyện QLoRA...")
    train_result = trainer.train()
    metrics = trainer.evaluate()

    # 6. Lưu Adapter cuối cùng
    final_adapter_dir = output_dir / "final_adapter"
    print(f"[*] Huấn luyện hoàn tất! Đang lưu Adapter vào: {final_adapter_dir}")
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    eval_loss = metrics.get("eval_loss", 0.0)
    perplexity = math.exp(min(eval_loss, 20))
    print("==========================================================")
    print(f"  KẾT QUẢ HUẤN LUYỆN:")
    print(f"  - Train Loss : {train_result.training_loss:.4f}")
    print(f"  - Eval Loss  : {eval_loss:.4f}")
    print(f"  - Perplexity : {perplexity:.4f}")
    print(f"  - Adapter    : {final_adapter_dir.resolve()}")
    print("==========================================================")


if __name__ == "__main__":
    main()
