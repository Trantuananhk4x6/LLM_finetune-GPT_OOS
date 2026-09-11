# Hướng Dẫn Huấn Luyện Qwen Smart Sales Trên Google Colab & Nền Tảng Cloud

Tài liệu này hướng dẫn chi tiết cách đưa toàn bộ mã nguồn, dữ liệu (Dataset) và mô hình (Model) lên **Google Colab (GPU T4 miễn phí / A100 / L4)**, **Kaggle** hoặc **RunPod / Vast.ai** để huấn luyện nhanh chóng và không tốn tài nguyên máy tính cá nhân.

---

## CÁCH 1: Chạy Trực Tiếp Bằng Git Clone (Khuyên Dùng)

Nếu bạn đã đẩy (push) mã nguồn lên GitHub/GitLab:

### Bước 1: Mở Google Colab
1. Truy cập [colab.research.google.com](https://colab.research.google.com/).
2. Chọn **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Chọn **T4 GPU** (hoặc L4 / A100 nếu dùng Colab Pro).

### Bước 2: Tạo các Cell và chạy theo thứ tự

#### **Cell 1: Tải mã nguồn & cài đặt thư viện**
```bash
# 1. Clone repository của bạn
!git clone https://github.com/<tai_khoan_cua_ban>/LLM_finetune-GPT_OOS.git
%cd LLM_finetune-GPT_OOS

# 2. Cài đặt các thư viện cần thiết
!pip install -q torch transformers datasets accelerate peft bitsandbytes safetensors pyyaml
!pip install -q -e .
```

#### **Cell 2: Sinh bộ dữ liệu tư vấn bán hàng thông minh (SPIN & FAB)**
```python
# Tạo sẵn 2.000 mẫu hội thoại chuẩn SFT
!python src/llm_finetune_gpt_oos/data/generate_smart_sales_dataset.py
```

#### **Cell 3: Khởi chạy huấn luyện QLoRA 4-bit**
```bash
# Huấn luyện trên Colab T4 (Chiếm < 9GB VRAM, siêu mượt)
!python scripts/train_cloud.py \
    --model_id "Qwen/Qwen2.5-7B-Instruct" \
    --train_file "data/processed/smart_sales_train.jsonl" \
    --val_file "data/processed/smart_sales_val.jsonl" \
    --output_dir "./artifacts/runs/qwen-smart-sales-cloud" \
    --max_seq_length 1536 \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 16
```
*(Nếu bạn muốn dùng bản Qwen 9B gốc, đổi `--model_id "Qwen/Qwen3.5-9B"`)*.

#### **Cell 4: Tải Adapter về Google Drive hoặc máy tính**
```python
from google.colab import drive, files
import shutil

# Nén thư mục adapter đã huấn luyện
shutil.make_archive("qwen_sales_adapter", "zip", "./artifacts/runs/qwen-smart-sales-cloud/final_adapter")

# Tải file zip về máy tính cá nhân
files.download("qwen_sales_adapter.zip")

# Hoặc lưu trực tiếp vào Google Drive của bạn:
# drive.mount('/content/drive')
# shutil.copy("qwen_sales_adapter.zip", "/content/drive/MyDrive/")
```

---

## CÁCH 2: Nạp Thư Mục Dự Án Từ Máy Tính Lên Google Drive

Nếu bạn chưa đưa code lên GitHub, bạn có thể bê nguyên thư mục dự án lên Google Drive:

1. **Nén thư mục dự án** (loại bỏ thư mục `.venv` và `cache` để file nén nhẹ chỉ vài MB).
2. **Kéo thả lên Google Drive** (ví dụ đặt tại thư mục `MyDrive/LLM_finetune-GPT_OOS`).
3. Mở Colab và chạy các lệnh sau:

```python
# 1. Kết nối Colab với Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Chuyển vào thư mục dự án trên Drive
%cd /content/drive/MyDrive/LLM_finetune-GPT_OOS

# 3. Cài thư viện và huấn luyện
!pip install -q torch transformers datasets accelerate peft bitsandbytes safetensors pyyaml
!pip install -q -e .

!python src/llm_finetune_gpt_oos/data/generate_smart_sales_dataset.py
!bash scripts/train_cloud.sh
```

---

## CÁCH 3: Huấn Luyện Trên Kaggle (30 giờ GPU T4/P100 miễn phí mỗi tuần)

1. Vào [kaggle.com/code](https://kaggle.com/code) $\rightarrow$ Tạo **New Notebook**.
2. Phía bên phải màn hình:
   - **Accelerator**: Chọn **GPU T4 x 2** hoặc **GPU P100**.
   - **Internet**: Bật **Internet on** (Bắt buộc để tải model Hugging Face).
3. Copy toàn bộ nội dung từ **Cách 1** dán vào notebook Kaggle và bấm **Run All**.

---

## ĐEM MODEL ĐÃ TRAIN VỀ DÙNG LOCAL TRÊN MÁY TÍNH (OLLAMA / VLLM)

Sau khi tải `final_adapter` từ Colab về máy cá nhân:
1. Giải nén vào thư mục `artifacts/runs/qwen35-smart-sales/final_adapter`.
2. Chạy lệnh merge trọng số và tạo GGUF:
```powershell
python scripts/export_and_quantize.py --base_model Qwen/Qwen2.5-7B-Instruct --adapter_path ./artifacts/runs/qwen35-smart-sales/final_adapter
```
3. Sau khi lượng tử hóa sang file `.gguf`, bạn có thể đưa thẳng vào **Ollama** để phục vụ tư vấn bán hàng với độ trễ cực thấp (< 0.2s)!
