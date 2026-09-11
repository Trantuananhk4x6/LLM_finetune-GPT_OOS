from typing import Any

import torch

from ..config import load_config
from ..models.base_model import load_adapter


class SalesGenerator:
    def __init__(self, config_path: str = "configs/inference.yaml"):
        self.config = load_config(config_path)
        self.model, self.tokenizer = load_adapter(self.config["model"], self.config["model"]["adapter_path"])

    def generate(self, prompt: str) -> dict[str, Any]:
        messages = [{"role": "system", "content": "Bạn là Sales Operations AI. Chỉ kết luận dựa trên dữ liệu người dùng cung cấp."}, {"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        options = self.config["generation"]
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=options["max_new_tokens"], do_sample=options["temperature"] > 0, temperature=options["temperature"], top_p=options["top_p"], repetition_penalty=options["repetition_penalty"], pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return {"response": self.tokenizer.decode(generated, skip_special_tokens=True).strip()}
