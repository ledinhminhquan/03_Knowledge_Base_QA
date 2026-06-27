"""Dense bi-encoder retriever + hybrid (dense ⊕ BM25 via RRF).

* Encodes the corpus into a FAISS vector store (``index/vector_store.py``).
* Applies the correct query-side convention per model family:
  - **bge**: prepend the query instruction; passages embedded raw.
  - **e5**:  ``query:`` / ``passage:`` prefixes.
  - **MiniLM/mpnet**: symmetric — no prefixes.
* Optionally fuses dense results with a BM25 sparse retriever via Reciprocal
  Rank Fusion (RRF) — reliably better on rare entities / OOD terms.

Heavy deps (sentence-transformers / faiss) are imported lazily so the package
imports without them; a small CPU fallback model is used if the primary fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import RetrieverConfig
from ..index.vector_store import Passage, VectorStore
from ..logging_utils import get_logger
from .model_registry import has_model, resolve_latest
from .baseline_bm25 import BM25Retriever

logger = get_logger(__name__)


class DenseEncoder:
    """sentence-transformers bi-encoder with family-aware prefixes."""

    def __init__(self, cfg: RetrieverConfig, model_dir: Optional[str] = None,
                 device: Optional[str] = None, prefer_fallback: bool = False):
        self.cfg = cfg
        self.device = device
        self._model = None
        self.name = None
        self._prefer_fallback = prefer_fallback
        self._explicit_dir = model_dir

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # prefer a fine-tuned checkpoint, else the base model, else the fallback
            candidate = None
            if self._explicit_dir:
                latest = resolve_latest(self._explicit_dir)
                if has_model(latest):
                    candidate = str(latest)
            if candidate is None:
                candidate = self.cfg.bi_encoder_fallback if self._prefer_fallback else self.cfg.bi_encoder_model
            try:
                self._model = SentenceTransformer(candidate, device=self.device)
                self.name = candidate
            except Exception as exc:
                logger.warning("Encoder %s failed (%s); using fallback %s", candidate, exc, self.cfg.bi_encoder_fallback)
                self._model = SentenceTransformer(self.cfg.bi_encoder_fallback, device=self.device)
                self.name = self.cfg.bi_encoder_fallback
        return self._model

    def _is_bge(self) -> bool:
        return "bge" in (self.name or self.cfg.bi_encoder_model).lower()

    def encode_queries(self, texts: Sequence[str]):
        prepared = list(texts)
        if self.cfg.e5_style:
            prepared = [f"query: {t}" for t in prepared]
        elif self._is_bge():
            prepared = [self.cfg.query_instruction + t for t in prepared]
        return self._encode(prepared)

    def encode_passages(self, texts: Sequence[str]):
        prepared = list(texts)
        if self.cfg.e5_style:
            prepared = [f"passage: {t}" for t in prepared]
        return self._encode(prepared)

    def _encode(self, texts: Sequence[str]):
        import numpy as np
        emb = self.model.encode(list(texts), batch_size=self.cfg.embed_batch_size,
                                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")


class Retriever:
    """Hybrid retriever over a corpus: dense FAISS ⊕ optional BM25 (RRF fusion)."""

    def __init__(self, cfg: Optional[RetrieverConfig] = None, model_dir: Optional[str] = None,
                 device: Optional[str] = None, prefer_fallback: bool = False):
        self.cfg = cfg or RetrieverConfig()
        self.encoder = DenseEncoder(self.cfg, model_dir=model_dir, device=device, prefer_fallback=prefer_fallback)
        self.store: Optional[VectorStore] = None
        self.bm25: Optional[BM25Retriever] = None

    # ---- build / persist ---------------------------------------------------
    def build(self, passages: Sequence[Passage]) -> "Retriever":
        passages = list(passages)
        try:  # dense index (degrade to BM25-only if the encoder is unavailable)
            emb = self.encoder.encode_passages([p.text for p in passages])
            self.store = VectorStore().build(emb, passages)
        except Exception as exc:
            logger.warning("Dense encoder unavailable (%s); BM25-only retrieval.", exc)
            self.store = None
            self._bm25_passages = passages
        if self.cfg.use_bm25 or self.store is None:
            self.bm25 = BM25Retriever().index(passages)
        logger.info("Retriever built over %d passages (dense=%s, bm25=%s)",
                    len(passages), self.store is not None, self.bm25 is not None)
        return self

    def add(self, passages: Sequence[Passage]) -> None:
        passages = list(passages)
        try:
            emb = self.encoder.encode_passages([p.text for p in passages])
            if self.store is None:
                self.store = VectorStore().build(emb, passages)
            else:
                self.store.add(emb, passages)
            all_p = self.store.passages
        except Exception as exc:
            logger.warning("Dense add failed (%s); BM25-only.", exc)
            existing = getattr(self, "_bm25_passages", [])
            all_p = list(existing) + passages
            self._bm25_passages = all_p
        if self.cfg.use_bm25 or self.store is None:
            self.bm25 = BM25Retriever().index(all_p)  # rebuild sparse index

    @property
    def size(self) -> int:
        if self.store is not None:
            return len(self.store)
        if self.bm25 is not None:
            return len(self.bm25.passages)
        return 0

    def save(self, dir_path: Optional[str] = None) -> Path:
        from ..config import index_dir
        d = Path(dir_path or index_dir())
        if self.store is not None:
            self.store.save(d)
        return d

    def load(self, dir_path: Optional[str] = None) -> "Retriever":
        from ..config import index_dir
        d = Path(dir_path or index_dir())
        self.store = VectorStore.load(d)
        if self.cfg.use_bm25:
            self.bm25 = BM25Retriever().index(self.store.passages)
        return self

    # ---- retrieve ----------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        top_k = top_k or self.cfg.top_k
        # BM25-only path (no dense index available)
        if self.store is None or not self.store.passages:
            if self.bm25 is not None:
                sparse = self.bm25.search(query, top_k=top_k)
                return [_to_dict(p, bm25_score=s, score=s, rank=i) for i, (p, s) in enumerate(sparse)]
            return []
        q = self.encoder.encode_queries([query])[0]
        dense = self.store.search(q, top_k=top_k)  # [(Passage, score)]
        if not (self.cfg.use_bm25 and self.bm25 is not None):
            return [_to_dict(p, dense_score=s, rank=i) for i, (p, s) in enumerate(dense)]
        sparse = self.bm25.search(query, top_k=top_k)
        return _rrf_fuse(dense, sparse, k=self.cfg.rrf_k, top_k=top_k)


def _to_dict(p: Passage, dense_score: float = 0.0, bm25_score: float = 0.0,
             score: Optional[float] = None, rank: int = 0) -> Dict:
    return {"id": p.id, "text": p.text, "title": p.title, "doc_id": p.doc_id,
            "dense_score": float(dense_score), "bm25_score": float(bm25_score),
            "score": float(score if score is not None else dense_score), "rank": rank}


def _rrf_fuse(dense: List[Tuple[Passage, float]], sparse: List[Tuple[Passage, float]],
              k: int = 60, top_k: int = 20) -> List[Dict]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank) across rankers."""
    records: Dict[str, Dict] = {}
    for rank, (p, s) in enumerate(dense):
        rec = records.setdefault(p.id, {"passage": p, "rrf": 0.0, "dense": 0.0, "bm25": 0.0})
        rec["rrf"] += 1.0 / (k + rank + 1)
        rec["dense"] = float(s)
    for rank, (p, s) in enumerate(sparse):
        rec = records.setdefault(p.id, {"passage": p, "rrf": 0.0, "dense": 0.0, "bm25": 0.0})
        rec["rrf"] += 1.0 / (k + rank + 1)
        rec["bm25"] = float(s)
    fused = sorted(records.values(), key=lambda r: r["rrf"], reverse=True)[:top_k]
    return [_to_dict(r["passage"], dense_score=r["dense"], bm25_score=r["bm25"], score=r["rrf"], rank=i)
            for i, r in enumerate(fused)]


__all__ = ["DenseEncoder", "Retriever"]
