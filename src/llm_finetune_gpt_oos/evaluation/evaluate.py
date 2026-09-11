import math
from typing import Any

from transformers import DataCollatorForSeq2Seq, Trainer

from ..config import load_config
from ..models.base_model import load_adapter
from ..models.trainer import ConversationDataset, build_training_arguments


def evaluate_model(config_path: str = "configs/train.yaml", adapter_path: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    training = config["training"]
    resolved_adapter = adapter_path or str(training["output_dir"] + "/final_adapter")
    model, tokenizer = load_adapter(config["model"], resolved_adapter)
    dataset = ConversationDataset(training["validation_file"], tokenizer, training["max_seq_length"], config["model"].get("cache_dir"))
    arguments = build_training_arguments({**training, "output_dir": str(training["output_dir"] + "/evaluation")})
    trainer = Trainer(model=model, args=arguments, eval_dataset=dataset, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8))
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss")
    return {"adapter_path": resolved_adapter, "validation_examples": len(dataset), "loss": loss, "perplexity": math.exp(min(loss, 20)) if loss is not None else None}
