from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(path: str | Path, root: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate
