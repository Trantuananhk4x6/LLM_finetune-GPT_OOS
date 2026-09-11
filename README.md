# Sales Operations LLM Fine-tuning

Hệ thống QLoRA có kiểm soát truy xuất nguồn dữ liệu để tinh chỉnh Qwen cho tác vụ phân tích bán hàng, vận hành đơn hàng và hỗ trợ khách hàng. Thiết kế tách biệt dữ liệu, adapter train, đánh giá và inference để adapter có thể được version hóa, thay thế hoặc rollback mà không đụng checkpoint nền.

## Kiến trúc

```text
Kaggle ZIP hoặc CSV
        │
        ▼
PII filter → schema adapter → chat SFT JSONL → split có seed → manifest + SHA-256
                                                        │
                                                        ▼
Qwen Hugging Face checkpoint → 4-bit NF4 → LoRA adapter → evaluation → final_adapter
                                                        │
                                                        ▼
                                      Qwen base + final_adapter → Sales Operations AI
```

`models/base/qwen3.5-9b-ollama-gguf/qwen3.5-9b-q4_k_m.gguf` là bản sao Qwen 3.5 9B cục bộ đang có trên máy, định dạng GGUF Q4_K_M của Ollama. Nó dùng cho inference local; QLoRA cần checkpoint Qwen gốc theo định dạng Hugging Face để có tokenizer, kiến trúc và trọng số trainable. Không thể fine-tune trực tiếp một GGUF lượng tử hoá bằng `transformers`/`peft`.

## Dữ liệu

Nguồn hiện đã tải vào dự án là [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail): 541.909 giao dịch của một nhà bán lẻ trực tuyến tại Anh, được phát hành theo CC BY 4.0. Pipeline đọc trực tiếp XLSX theo streaming và lấy mẫu ngẫu nhiên có seed, nên không cần chuyển đổi thủ công hoặc nạp toàn bộ dataset vào RAM.

Đặt file ZIP, CSV hoặc XLSX vào `data/raw/`. Pipeline tự:

- kiểm tra đường dẫn ZIP trước khi giải nén;
- loại các cột có dấu hiệu PII như tên, email, điện thoại, địa chỉ và toạ độ;
- biến mỗi bản ghi thành conversation tiếng Việt theo chat format;
- chỉ huấn luyện loss trên phần trả lời của assistant;
- tạo `train.jsonl`, `validation.jsonl` và `manifest.json` chứa checksum, schema và số dòng nguồn.

`data/raw/vietnamese_sales_playbook.csv` là 17 kịch bản gốc tiếng Việt cho nội thất và điện máy: khám phá nhu cầu, ngân sách, so sánh, giao lắp, trả góp, bảo hành, đổi trả và hàng hết. Đây là nội dung mẫu tự tạo, không sao chép catalogue hay hội thoại của bất kỳ nhà bán lẻ nào.

Không dùng trực tiếp PII thật cho training. Dataset giao dịch chỉ giúp model học cách trình bày, kiểm tra giả định và đề xuất hành động; số liệu live phải do tool hoặc database đã phân quyền cung cấp lúc inference.

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

QLoRA 4-bit yêu cầu NVIDIA CUDA, PyTorch CUDA và VRAM đủ cho Qwen 9B cùng sequence length đã cấu hình. Cấu hình mặc định dùng micro-batch 1, gradient accumulation 16 và context 2048 để giảm áp lực VRAM. Model card Qwen yêu cầu Transformers bản mới nhất, vì vậy `requirements.txt` lấy trực tiếp branch `main` của Transformers.

## Vận hành

```powershell
.\scripts\prepare_data.ps1
.\scripts\generate_sales_playbook.ps1
.\scripts\train.ps1
.\scripts\evaluate.ps1
.\scripts\infer.ps1 'Phân tích dữ liệu: {"Product_Category":"Electronics","Quantity":4,"Unit_Price":120,"Discount_Applied":10}'
```

`generate_sales_playbook.ps1` tạo 1.000.000 mẫu synthetic tiếng Việt, tách xác định theo hash thành train và validation. Tập này phủ nội thất và điện máy, gồm discovery, tư vấn theo ngân sách, so sánh, chi phí sử dụng, tương thích, trả góp minh bạch, giao-lắp, đổi trả, bảo hành, tồn kho và mua doanh nghiệp. Các mẫu bắt buộc tôn trọng quyền quyết định, không khan hiếm giả, không phí ẩn và không gây áp lực.

Artifacts của mỗi lần train nằm trong `artifacts/runs/qwen35-sales-qlora/`:

- `checkpoint-*`: checkpoint adapter có thể resume;
- `final_adapter/`: adapter PEFT và tokenizer để inference;
- `training_summary.json`: base model, sample count, loss, perplexity và GPU;
- `data/processed/manifest.json`: lineage của tập dữ liệu.

## Cấu hình

- `configs/dataset.yaml`: đường dẫn dữ liệu, seed, sampling, PII filtering và nguồn.
- `configs/train.yaml`: Qwen base checkpoint, NF4, LoRA targets và hyperparameters.
- `configs/inference.yaml`: base checkpoint, adapter và generation policy.

Để train một phiên bản Qwen khác, chỉ thay `model.base_model_id` trong hai file train/inference bằng model Hugging Face tương ứng. Giữ base model của adapter và inference giống hệt nhau.
