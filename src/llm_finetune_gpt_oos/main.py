import argparse
import json

from .utils.logging_utils import configure_logging


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sales-finetune")
    parser.add_argument("command", choices=["prepare-data", "generate-sales-playbook", "generate-smart-sales", "train", "evaluate", "infer"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--adapter-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    logger = configure_logging("logs")
    if args.command == "prepare-data":
        from .data.prepare_data import prepare_data

        result = prepare_data(args.config or "configs/dataset.yaml")
    elif args.command == "generate-sales-playbook":
        from .data.generate_sales_playbook import generate_sales_playbook

        result = generate_sales_playbook(args.config or "configs/synthetic_sales.yaml")
    elif args.command == "generate-smart-sales":
        from .data.generate_smart_sales_dataset import generate_dataset

        generate_dataset()
        result = {"status": "ok"}
    elif args.command == "train":
        from .models.trainer import SalesQLoRATrainer

        result = SalesQLoRATrainer(args.config or "configs/train.yaml").train()
    elif args.command == "evaluate":
        from .evaluation.evaluate import evaluate_model

        result = evaluate_model(args.config or "configs/train.yaml", args.adapter_path)
    else:
        if not args.prompt:
            raise ValueError("--prompt is required for infer")
        from .inference.generate import SalesGenerator

        result = SalesGenerator(args.config or "configs/inference.yaml").generate(args.prompt)
    logger.info("command=%s completed", args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
