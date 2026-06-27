"""Download datasets + build the demo knowledge-base index (no large data committed)."""

from __future__ import annotations

from typing import Optional

from ..config import AppConfig, ensure_dirs
from ..logging_utils import get_logger

logger = get_logger(__name__)


def download_reader(cfg: AppConfig) -> dict:
    from .dataset import load_squad
    ds = load_squad(cfg.data)
    return {s: len(ds[s]) for s in ds}


def download_retriever(cfg: AppConfig) -> dict:
    from .dataset import load_retriever_pairs
    ds, q, p = load_retriever_pairs(cfg.data)
    return {"pairs": len(ds), "query_col": q, "passage_col": p}


def download_demo_kb(cfg: AppConfig, limit_corpus: Optional[int] = None) -> dict:
    from .corpus import build_demo_kb
    retriever = build_demo_kb(cfg, limit_corpus=limit_corpus)
    return {"passages": len(retriever.store) if retriever.store else 0}


def download_task(task: str, cfg: Optional[AppConfig] = None) -> dict:
    ensure_dirs()
    cfg = cfg or AppConfig()
    if task == "reader":
        return download_reader(cfg)
    if task == "retriever":
        return download_retriever(cfg)
    if task in ("corpus", "demo-kb", "demo_kb"):
        return download_demo_kb(cfg)
    raise ValueError(f"Unknown task: {task}")


def download_all(cfg: Optional[AppConfig] = None) -> dict:
    cfg = cfg or AppConfig()
    out = {}
    for task in ("reader", "retriever", "corpus"):
        try:
            out[task] = download_task(task, cfg)
        except Exception as exc:
            logger.warning("Download failed for %s: %s", task, exc)
            out[task] = {"error": str(exc)}
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["all", "reader", "retriever", "corpus"], default="all")
    a = ap.parse_args()
    res = download_all() if a.task == "all" else download_task(a.task)
    print(json.dumps(res, indent=2))
