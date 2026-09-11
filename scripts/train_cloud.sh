#!/bin/bash
# ==============================================================================
# Script tự động chạy huấn luyện Qwen Smart Sales trên Google Colab / Kaggle / RunPod
# ==============================================================================
set -e

echo "=== [1/4] Cài đặt các thư viện cần thiết ==="
pip install -q torch transformers datasets accelerate peft bitsandbytes safetensors pyyaml

echo "=== [2/4] Cài đặt repo ở chế độ editable ==="
pip install -q -e .

echo "=== [3/4] Sinh dữ liệu huấn luyện tư vấn bán hàng thông minh ==="
python src/llm_finetune_gpt_oos/data/generate_smart_sales_dataset.py

echo "=== [4/4] Bắt đầu huấn luyện QLoRA 4-bit trên GPU Cloud ==="
python scripts/train_cloud.py \
    --model_id "Qwen/Qwen2.5-7B-Instruct" \
    --train_file "data/processed/smart_sales_train.jsonl" \
    --val_file "data/processed/smart_sales_val.jsonl" \
    --output_dir "./artifacts/runs/qwen-smart-sales-cloud" \
    --max_seq_length 1536 \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 16 \
    --lr 2e-4

echo "=== Hoàn tất huấn luyện! Adapter đã được lưu tại ./artifacts/runs/qwen-smart-sales-cloud/final_adapter ==="
