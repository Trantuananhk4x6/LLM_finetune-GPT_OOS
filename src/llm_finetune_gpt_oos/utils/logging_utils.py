import logging
from pathlib import Path


def configure_logging(log_dir: str) -> logging.Logger:
    destination = Path(log_dir)
    destination.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sales_finetune")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = logging.FileHandler(destination / "run.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
