"""
Script Merge LoRA Adapter vào Base Model và chuẩn bị lượng tử hóa (Quantization)
Giúp mô hình nhẹ hơn (từ 18GB xuống còn ~5GB), chạy mượt trên Ollama / vLLM / LM Studio.
"""
import argparse
import os
import subprocess
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_export(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    torch_dtype: str = "bfloat16"
):
    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
    print(f"[*] Bắt đầu nạp Base Model: {base_model_id} (Dtype: {torch_dtype})")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="cpu",  # Merge trên RAM để tránh tràn VRAM
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    print(f"[*] Gắn LoRA Adapter từ: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("[*] Đang thực hiện Merge LoRA vào Base Model weights...")
    merged_model = model.merge_and_unload()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Lưu mô hình sau khi gộp vào: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print("[OK] Đã merge xong mô hình hoàn chỉnh!")
    print("\n--- HƯỚNG DẪN XUẤT SANG ĐỊNH DẠNG GGUF ĐỂ SIÊU NHẸ (CHẠY OLLAMA / LLAMA.CPP) ---")
    print("1. Cài đặt llama.cpp:")
    print("   git clone https://github.com/ggerganov/llama.cpp")
    print("   pip install -r llama.cpp/requirements.txt")
    print(f"2. Chuyển đổi sang FP16 GGUF:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {output_dir} --outtype f16 --outfile {output_dir}/qwen_sales_f16.gguf")
    print("3. Nén lượng tử hóa xuống 4-bit (Q4_K_M) cực nhẹ (~5.5 GB VRAM):")
    print(f"   llama.cpp/llama-quantize {output_dir}/qwen_sales_f16.gguf {output_dir}/qwen_sales_q4_k_m.gguf Q4_K_M")
    print("\n4. Chạy trực tiếp trên Ollama bằng Modelfile:")
    print(f"   FROM {output_dir}/qwen_sales_q4_k_m.gguf")
    print("   PARAMETER temperature 0.7")
    print("   PARAMETER top_p 0.9")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA and prepare for quantization")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-9B", help="Base model HuggingFace ID")
    parser.add_argument("--adapter_path", type=str, default="./artifacts/runs/qwen35-sales-qlora/final_adapter", help="Path to LoRA adapter")
    parser.add_argument("--output_dir", type=str, default="./models/merged_qwen_sales", help="Directory to save merged model")
    args = parser.parse_args()

    merge_and_export(args.base_model, args.adapter_path, args.output_dir)
