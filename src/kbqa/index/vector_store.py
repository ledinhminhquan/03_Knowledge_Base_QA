"""FAISS-backed vector store for the document knowledge base.

Stores normalised passage embeddings in a FAISS inner-product index (cosine
similarity) alongside the passage records (id, text, metadata), and persists both
to disk so the KB survives restarts. If FAISS is not installed it transparently
falls back to a NumPy brute-force search, so the system always runs.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Passage:
    """One indexed chunk of a knowledge-base document."""
    id: str
    text: str
    doc_id: str = ""
    title: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "doc_id": self.doc_id, "title": self.title, "meta": self.meta}


class VectorStore:
    """Dense vector index over passages with cosine similarity search."""

    def __init__(self, dim: Optional[int] = None):
        self.dim = dim
        self._index = None              # faiss index (or None → numpy fallback)
        self._matrix = None             # numpy fallback matrix [n, dim]
        self.passages: List[Passage] = []
        self._use_faiss = False

    # ---- building ----------------------------------------------------------
    def _new_faiss(self, dim: int):
        try:
            import faiss
            self._use_faiss = True
            return faiss.IndexFlatIP(dim)
        except Exception as exc:
            logger.info("faiss unavailable (%s); using NumPy brute-force search.", exc)
            self._use_faiss = False
            return None

    def build(self, embeddings, passages: Sequence[Passage]) -> "VectorStore":
        import numpy as np

        emb = np.ascontiguousarray(np.asarray(embeddings, dtype="float32"))
        if emb.ndim != 2:
            raise ValueError("embeddings must be 2-D [n, dim]")
        self.dim = emb.shape[1]
        self.passages = list(passages)
        if len(self.passages) != emb.shape[0]:
            raise ValueError("number of passages must match number of embeddings")

        self._index = self._new_faiss(self.dim)
        if self._index is not None:
            self._index.add(emb)
        else:
            self._matrix = emb
        logger.info("Built vector store: %d passages, dim=%d, faiss=%s",
                    len(self.passages), self.dim, self._use_faiss)
        return self

    def add(self, embeddings, passages: Sequence[Passage]) -> None:
        """Incrementally add passages (supports KB ingestion at runtime)."""
        import numpy as np

        emb = np.ascontiguousarray(np.asarray(embeddings, dtype="float32"))
        if self._index is None and self._matrix is None:
            self.build(emb, passages)
            return
        if self._use_faiss and self._index is not None:
            self._index.add(emb)
        else:
            self._matrix = np.vstack([self._matrix, emb]) if self._matrix is not None else emb
        self.passages.extend(passages)

    # ---- search ------------------------------------------------------------
    def search(self, query_emb, top_k: int = 10) -> List[Tuple[Passage, float]]:
        import numpy as np

        if not self.passages:
            return []
        q = np.ascontiguousarray(np.asarray(query_emb, dtype="float32").reshape(1, -1))
        top_k = min(top_k, len(self.passages))
        if self._use_faiss and self._index is not None:
            scores, idxs = self._index.search(q, top_k)
            pairs = [(self.passages[int(i)], float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
        else:
            sims = (self._matrix @ q[0])
            order = np.argsort(sims)[::-1][:top_k]
            pairs = [(self.passages[int(i)], float(sims[int(i)])) for i in order]
        return pairs

    # ---- persistence -------------------------------------------------------
    def save(self, dir_path: str | Path) -> Path:
        import numpy as np

        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"dim": self.dim, "use_faiss": self._use_faiss, "n": len(self.passages)}
        (d / "store_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        with (d / "passages.pkl").open("wb") as fh:
            pickle.dump([p.to_dict() for p in self.passages], fh)
        if self._use_faiss and self._index is not None:
            import faiss
            faiss.write_index(self._index, str(d / "index.faiss"))
        elif self._matrix is not None:
            np.save(d / "matrix.npy", self._matrix)
        logger.info("Saved vector store -> %s", d)
        return d

    @classmethod
    def load(cls, dir_path: str | Path) -> "VectorStore":
        import numpy as np

        d = Path(dir_path)
        meta = json.loads((d / "store_meta.json").read_text(encoding="utf-8"))
        store = cls(dim=meta["dim"])
        with (d / "passages.pkl").open("rb") as fh:
            store.passages = [Passage(**rec) for rec in pickle.load(fh)]
        if meta.get("use_faiss") and (d / "index.faiss").exists():
            import faiss
            store._index = faiss.read_index(str(d / "index.faiss"))
            store._use_faiss = True
        elif (d / "matrix.npy").exists():
            store._matrix = np.load(d / "matrix.npy")
            store._use_faiss = False
        logger.info("Loaded vector store (%d passages) <- %s", len(store.passages), d)
        return store

    def __len__(self) -> int:
        return len(self.passages)


__all__ = ["VectorStore", "Passage"]
