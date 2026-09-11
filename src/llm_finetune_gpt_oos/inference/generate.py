from typing import Any

import torch

from ..config import load_config
from ..models.base_model import load_adapter


SALES_SYSTEM_PROMPT = (
    "Bạn là chuyên viên tư vấn bán hàng xuất sắc của hệ thống Điện Máy & Nội Thất Gia Đình. "
    "Phong cách: (1) Xưng hô lễ phép, thân thiện, chu đáo (Dạ em chào anh/chị...). "
    "(2) Luôn hỏi rõ nhu cầu (diện tích, số người, thói quen, ngân sách) trước khi chốt mẫu. "
    "(3) Giải thích giá trị bằng công năng thực tế (FAB: Đặc điểm → Ưu điểm → Lợi ích cho gia đình), "
    "không đọc thông số khô khan. "
    "(4) Xử lý từ chối khéo léo, không ép khách nhưng luôn tạo lý do hấp dẫn để chốt sớm. "
    "(5) Trung thực về giá niêm yết, chính sách bảo hành chính hãng và điều kiện giao lắp thực tế. "
    "(6) Tôn trọng quyền quyết định của khách, không dùng khan hiếm giả và không che giấu chi phí."
)


class SalesGenerator:
    def __init__(self, config_path: str = "configs/smart_sales_inference.yaml"):
        self.config = load_config(config_path)
        self.model, self.tokenizer = load_adapter(
            self.config["model"], self.config["model"]["adapter_path"]
        )

    def generate(self, prompt: str, catalog_context: str | None = None) -> dict[str, Any]:
        """
        Sinh câu trả lời tư vấn bán hàng.

        Args:
            prompt: Câu hỏi của khách hàng.
            catalog_context: (Tùy chọn) Thông tin sản phẩm/kho hàng dạng JSON
                             để AI trả lời chính xác giá và tồn kho.
        """
        system_content = SALES_SYSTEM_PROMPT
        if catalog_context:
            system_content += (
                "\n\nDỮ LIỆU KHO HÀNG THỰC TẾ HIỆN CÓ (chỉ báo giá và tồn kho "
                "dựa trên dữ liệu này, không bịa):\n" + catalog_context
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        options = self.config["generation"]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=options["max_new_tokens"],
                do_sample=options["temperature"] > 0,
                temperature=options["temperature"],
                top_p=options["top_p"],
                repetition_penalty=options["repetition_penalty"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return {"response": self.tokenizer.decode(generated, skip_special_tokens=True).strip()}
