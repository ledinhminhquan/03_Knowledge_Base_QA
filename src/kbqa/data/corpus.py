"""Knowledge-base construction: documents → chunks → embedded FAISS index.

Turns a set of documents into a persisted, searchable knowledge base using the
chunker + retriever. Used by ``kbqa data --task corpus`` and the API ``/ingest``
endpoint. Idempotent: re-chunking the same docs yields the same passage ids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..config import AppConfig, index_dir
from ..logging_utils import get_logger
from .chunking import chunk_corpus
from .samples import SAMPLE_DOCS

logger = get_logger(__name__)


def build_kb_from_docs(cfg: AppConfig, documents: Sequence[Dict], save: bool = True,
                       index_path: Optional[str] = None):
    """Chunk + embed + index a list of {id?, title?, text} documents."""
    from ..models.retriever import Retriever

    passages = chunk_corpus(documents, chunk_size=cfg.chunk.chunk_size_words,
                            overlap=cfg.chunk.overlap_words)
    passages = [p for p in passages if len(p.text.split()) >= cfg.chunk.min_words] or passages
    retriever = Retriever(cfg.retriever, model_dir=str(cfg.retriever.output_dir)).build(passages)
    if save:
        retriever.save(index_path or str(index_dir()))
        _write_manifest(cfg, len(passages), index_path)
    logger.info("Built KB: %d docs -> %d passages", len(documents), len(passages))
    return retriever


def build_demo_kb(cfg: AppConfig, limit_corpus: Optional[int] = None, use_dataset: bool = True):
    """Build the demo KB from rag-mini-wikipedia (or the built-in samples)."""
    docs: List[Dict]
    if use_dataset:
        try:
            from .dataset import load_demo_kb
            docs, _ = load_demo_kb(cfg.data, limit_corpus=limit_corpus)
        except Exception as exc:
            logger.warning("Demo KB dataset unavailable (%s); using built-in samples.", exc)
            docs = list(SAMPLE_DOCS)
    else:
        docs = list(SAMPLE_DOCS)
    return build_kb_from_docs(cfg, docs)


def load_kb(cfg: AppConfig, index_path: Optional[str] = None):
    """Load a previously built KB index."""
    from ..models.retriever import Retriever
    return Retriever(cfg.retriever, model_dir=str(cfg.retriever.output_dir)).load(index_path or str(index_dir()))


def _write_manifest(cfg: AppConfig, n_passages: int, index_path: Optional[str]) -> None:
    import json
    from ..logging_utils import utc_now_iso

    d = Path(index_path or index_dir())
    manifest = {"model_version": cfg.serving.model_version, "encoder": cfg.retriever.bi_encoder_model,
                "n_passages": n_passages, "chunk_size_words": cfg.chunk.chunk_size_words,
                "built_at": utc_now_iso()}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


__all__ = ["build_kb_from_docs", "build_demo_kb", "load_kb"]
