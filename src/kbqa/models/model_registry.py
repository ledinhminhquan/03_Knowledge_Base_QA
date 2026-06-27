"""Model versioning + metadata utilities (shared by retriever and reader)."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from ..logging_utils import get_logger, utc_now_iso

logger = get_logger(__name__)

METADATA_FILE = "model_metadata.json"


def resolve_latest(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    latest = base / "latest"
    if latest.exists():
        return latest
    if base.exists():
        subdirs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
        if subdirs:
            return subdirs[-1]
    return base


def has_model(model_dir: str | Path) -> bool:
    d = Path(model_dir)
    return any((d / f).exists() for f in ("config.json", "model.safetensors", "pytorch_model.bin", "modules.json"))


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def environment_snapshot() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "datasets": _pkg_version("datasets"),
        "sentence_transformers": _pkg_version("sentence-transformers"),
        "faiss": _pkg_version("faiss-cpu") or _pkg_version("faiss-gpu"),
        "git_sha": git_sha(),
    }


def save_model_metadata(model_dir: str | Path, *, base_model: str, task: str,
                        config_subset: Dict[str, Any], dataset_info: Dict[str, Any],
                        metrics: Optional[Dict[str, Any]] = None, version: str = "1.0.0") -> Path:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": utc_now_iso(), "task": task, "base_model": base_model, "version": version,
        "config": config_subset, "dataset": dataset_info, "metrics": metrics or {},
        "environment": environment_snapshot(),
    }
    path = d / METADATA_FILE
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote model metadata -> %s", path)
    return path


def load_model_metadata(model_dir: str | Path) -> Dict[str, Any]:
    path = Path(model_dir) / METADATA_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


__all__ = ["resolve_latest", "has_model", "save_model_metadata", "load_model_metadata",
           "environment_snapshot", "git_sha", "METADATA_FILE"]
