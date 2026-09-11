"""
Sales Agent Module: Tích hợp mô hình Qwen đã tinh chỉnh với danh mục sản phẩm (Product Catalog)
Đảm bảo AI:
1. Tư vấn khôn khéo, thấu cảm, chốt deal nhanh (nhờ Fine-tuned QLoRA weights).
2. Nói đúng giá, đúng khuyến mãi, đúng tồn kho 100% (nhờ Catalog Injection / RAG nhẹ).
"""
import json
from pathlib import Path
from typing import Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Danh mục mẫu các sản phẩm Điện máy & Nội thất đang kinh doanh (Catalog Database)
SAMPLE_PRODUCT_CATALOG = [
    {
        "id": "DM-DH-01",
        "category": "Điều hòa",
        "name": "Điều hòa Daikin Inverter 1.5 HP FTKF35XVMV",
        "price": 11990000,
        "old_price": 13490000,
        "promotions": ["Tặng 100% công lắp đặt", "Tặng 5m ống đồng Thái Lan", "Trả góp 0%"],
        "stock": 15,
        "specs": "Công suất 12.000 BTU, diện tích 15-20m2, phin lọc Enzyme Blue diệt khuẩn 99.9%, luồng gió Coanda không phả vào người."
    },
    {
        "id": "DM-DH-02",
        "category": "Điều hòa",
        "name": "Điều hòa Casper Inverter 1.5 HP TC-12IS36",
        "price": 6890000,
        "old_price": 7990000,
        "promotions": ["Miễn phí công lắp đặt", "Bảo hành 1 đổi 1 trong 1 năm"],
        "stock": 28,
        "specs": "Công suất 12.000 BTU, làm lạnh siêu tốc Turbo trong 30 giây, dàn tản nhiệt mạ vàng chống ăn mòn."
    },
    {
        "id": "DM-TL-01",
        "category": "Tủ lạnh",
        "name": "Tủ lạnh Panasonic Inverter 322 lít NR-BV360QSVN",
        "price": 9890000,
        "old_price": 11500000,
        "promotions": ["Tặng nồi chiên không dầu Lock&Lock trị giá 1.500.000đ", "Miễn phí vận chuyển"],
        "stock": 8,
        "specs": "Ngăn đông mềm Prime Fresh+ -3 độ C giữ thịt cá tươi ngon 7 ngày không rã đông, Ag Clean kháng khuẩn khử mùi."
    },
    {
        "id": "NT-SF-01",
        "category": "Sofa",
        "name": "Sofa Góc L Vải Nano Chống Mèo Cào Kháng Nước Nordic Luxury",
        "price": 12900000,
        "old_price": 16000000,
        "promotions": ["Tặng 3 gối ôm đồng màu + 1 đôn phụ", "Giao lắp miễn phí tận phòng"],
        "stock": 5,
        "specs": "Kích thước 2m6 x 1m6, nệm mút D40 êm ái chống xẹp, khung gỗ sồi tự nhiên chống mối mọt bảo hành 5 năm."
    },
    {
        "id": "NT-BA-01",
        "category": "Bàn ăn",
        "name": "Bộ Bàn Ăn Thông Minh Mặt Đá Ceramic Mở Rộng 1m3 - 1m8 + 6 Ghế",
        "price": 8900000,
        "old_price": 11900000,
        "promotions": ["Giảm trực tiếp 200k khi thanh toán ngay", "Tặng gói bảo dưỡng đánh bóng mặt bàn"],
        "stock": 12,
        "specs": "Mặt đá Ceramic chống trầy, chịu nhiệt 1200 độ C, khung chân thép carbon sơn tĩnh điện chịu lực 200kg."
    }
]


class SmartSalesAgent:
    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen3.5-9B",
        adapter_path: Optional[str] = "./artifacts/runs/qwen35-sales-qlora/final_adapter",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.catalog = SAMPLE_PRODUCT_CATALOG
        print(f"[*] Đang tải Tokenizer từ: {base_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

        print(f"[*] Đang tải mô hình trên thiết bị: {device}")
        # Dùng 4-bit nếu ở trên GPU có bitsandbytes, ngược lại fallback bfloat16/float16
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )

        # Nếu có trained adapter thì load thêm PEFT adapter
        if adapter_path and Path(adapter_path).exists():
            print(f"[*] Gắn PEFT Adapter từ: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

    def find_relevant_products(self, query: str) -> list[dict[str, Any]]:
        """Tìm kiếm các sản phẩm trong kho phù hợp với câu hỏi của khách (Mô phỏng RAG / Search Tool)."""
        query_lower = query.lower()
        matched = []
        for p in self.catalog:
            if (
                p["category"].lower() in query_lower or
                any(word in query_lower for word in p["name"].lower().split()) or
                p["category"].lower() in ["điều hòa", "sofa", "tủ lạnh", "bàn ăn"]
            ):
                matched.append(p)
        return matched[:3] if matched else self.catalog[:2]

    def build_prompt(self, user_query: str, chat_history: list[dict[str, str]] = None) -> list[dict[str, str]]:
        """Ghép thông tin tồn kho thực tế vào ngữ cảnh để AI tư vấn chuẩn xác 100%."""
        relevant_products = self.find_relevant_products(user_query)
        catalog_info = json.dumps(relevant_products, ensure_ascii=False, indent=2)

        system_instruction = f"""Bạn là chuyên viên tư vấn bán hàng số 1 của chuỗi Điện Máy & Nội Thất Gia Đình.
DỮ LIỆU KHO HÀNG VÀ CHƯƠNG TRÌNH KHUYẾN MÃI THỰC TẾ HIỆN CÓ:
{catalog_info}

NGUYÊN TẮC TƯ VẤN:
1. Tư vấn thấu cảm, khôn khéo, phân tích đúng nhu cầu (diện tích, đối tượng dùng) và nêu bật lợi ích thực tế (tiết kiệm điện, bảo vệ sức khỏe, độ bền lâu dài).
2. Tuyệt đối báo đúng giá bán và quà tặng niêm yết trong kho hàng ở trên.
3. Luôn đưa ra câu hỏi gợi mở hoặc kỹ thuật chốt 2 chọn 1 để hướng khách hàng đến quyết định mua hàng."""

        messages = [{"role": "system", "content": system_instruction}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_query})
        return messages

    def respond(self, user_query: str, chat_history: list[dict[str, str]] = None) -> str:
        messages = self.build_prompt(user_query, chat_history)
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()


if __name__ == "__main__":
    print("[+] Khởi chạy SmartSalesAgent Demo...")
    # Demo thử prompt logic
    agent = SmartSalesAgent(base_model_id="Qwen/Qwen3.5-9B", adapter_path=None)
    test_query = "Phòng ngủ 16m2 có nắng chiều chiếu vào thì lắp điều hòa nào tốt em?"
    print(f"\nKhách hỏi: {test_query}")
    answer = agent.respond(test_query)
    print(f"\nAI Tư vấn:\n{answer}")
