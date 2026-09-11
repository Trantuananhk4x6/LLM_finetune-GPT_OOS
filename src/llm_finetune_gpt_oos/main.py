import argparse
import json

from .data.prepare_data import prepare_data
from .evaluation.evaluate import evaluate_model
from .inference.generate import SalesGenerator
from .models.trainer import SalesQLoRATrainer
from .utils.logging_utils import configure_logging


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sales-finetune")
    parser.add_argument("command", choices=["prepare-data", "train", "evaluate", "infer"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--adapter-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    logger = configure_logging("logs")
    if args.command == "prepare-data":
        result = prepare_data(args.config or "configs/dataset.yaml")
    elif args.command == "train":
        result = SalesQLoRATrainer(args.config or "configs/train.yaml").train()
    elif args.command == "evaluate":
        result = evaluate_model(args.config or "configs/train.yaml", args.adapter_path)
    else:
        if not args.prompt:
            raise ValueError("--prompt is required for infer")
        result = SalesGenerator(args.config or "configs/inference.yaml").generate(args.prompt)
    logger.info("command=%s completed", args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
