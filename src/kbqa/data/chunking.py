"""Document chunking for the knowledge base.

Splits documents into overlapping passages so each chunk fits the retriever's
context window and keeps enough local context for grounded answering. Overlap
preserves sentences that straddle a boundary.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional

from ..index.vector_store import Passage
from .preprocessing import clean_text

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> List[str]:
    """Split ``text`` into overlapping word windows of ~``chunk_size`` words.

    Word-based windows are model-agnostic and predictable; ``overlap`` words are
    repeated between consecutive chunks to avoid cutting context mid-thought.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_document(
    text: str,
    doc_id: Optional[str] = None,
    title: str = "",
    chunk_size: int = 180,
    overlap: int = 40,
    meta: Optional[dict] = None,
) -> List[Passage]:
    """Turn one document into a list of :class:`Passage` chunks."""
    text = clean_text(text)
    doc_id = doc_id or _hash_id(title, text[:200])
    passages: List[Passage] = []
    for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
        pid = f"{doc_id}::{i}"
        passages.append(Passage(id=pid, text=chunk, doc_id=doc_id, title=title, meta=meta or {}))
    return passages


def chunk_corpus(documents, chunk_size: int = 180, overlap: int = 40) -> List[Passage]:
    """Chunk an iterable of {id?, title?, text} dicts into passages."""
    out: List[Passage] = []
    for d in documents:
        out.extend(chunk_document(
            d.get("text", ""), doc_id=d.get("id"), title=d.get("title", ""),
            chunk_size=chunk_size, overlap=overlap, meta=d.get("meta"),
        ))
    return out


__all__ = ["chunk_text", "chunk_document", "chunk_corpus"]
