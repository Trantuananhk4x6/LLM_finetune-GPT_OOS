"""Reproducibility envelope: mọi run phải tự mô tả được chính nó.

Một checkpoint không kèm theo git SHA, hash dataset, phiên bản thư viện và seed
thì không tái lập được, và do đó không dùng được để so sánh hai lần train.
Module này sinh ra `run_manifest.json` đi kèm mỗi artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "trl", "numpy")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_object(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_state() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    status = _git("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_files": [line[3:] for line in (status or "").splitlines()][:50],
    }


def package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_state() -> dict[str, Any]:
    state: dict[str, Any] = {"platform": platform.platform(), "python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    try:
        import torch
    except ImportError:
        state["torch"] = None
        return state
    state["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        state["gpu_name"] = properties.name
        state["gpu_total_memory_gb"] = round(properties.total_memory / 1024**3, 2)
        state["gpu_capability"] = f"{properties.major}.{properties.minor}"
        state["gpu_count"] = torch.cuda.device_count()
        state["bf16_supported"] = torch.cuda.is_bf16_supported()
    return state


def set_determinism(seed: int, strict: bool = False) -> None:
    """Seed mọi nguồn ngẫu nhiên. `strict` đánh đổi tốc độ lấy tính lặp lại bit-exact."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            pass


def build_run_manifest(
    *,
    config: dict[str, Any],
    config_path: str | Path | None = None,
    data_files: dict[str, str | Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ảnh chụp đầy đủ của một run: code, dữ liệu, môi trường, cấu hình."""
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "packages": package_versions(),
        "hardware": hardware_state(),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": sha256_object(config),
        "config": config,
    }
    if data_files:
        manifest["data"] = {
            name: ({"path": str(path), "sha256": sha256_file(path), "bytes": Path(path).stat().st_size} if Path(path).exists() else {"path": str(path), "missing": True})
            for name, path in data_files.items()
        }
    if extra:
        manifest.update(extra)
    return manifest


def write_run_manifest(destination: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
