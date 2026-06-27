"""Cross-encoder reranker — the highest-ROI accuracy lever in RAG.

Reranks the retriever's top-k candidates by jointly scoring each (query, passage)
pair, then keeps the top-n. CPU default is the tiny ``ms-marco-MiniLM-L-6-v2``;
a GPU config flag swaps in the stronger ``bge-reranker-v2-m3``. Degrades to a
no-op (keeps retriever order) if sentence-transformers is unavailable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..config import RerankerConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class Reranker:
    def __init__(self, cfg: Optional[RerankerConfig] = None, device: Optional[str] = None):
        self.cfg = cfg or RerankerConfig()
        self.device = device
        self._model = None
        self.name = None
        self._failed = False

    @property
    def model(self):
        if self._model is None and not self._failed:
            from sentence_transformers import CrossEncoder

            name = self.cfg.cross_encoder_gpu if self.cfg.use_gpu_reranker else self.cfg.cross_encoder_model
            try:
                self._model = CrossEncoder(name, max_length=self.cfg.max_length, device=self.device)
                self.name = name
            except Exception as exc:
                logger.warning("Reranker %s unavailable (%s); reranking disabled.", name, exc)
                self._failed = True
        return self._model

    def rerank(self, query: str, candidates: Sequence[Dict], top_n: Optional[int] = None) -> List[Dict]:
        top_n = top_n or self.cfg.rerank_top_n
        cands = list(candidates)
        if not cands:
            return []
        model = self.model
        if model is None:  # graceful no-op
            for i, c in enumerate(cands[:top_n]):
                c["rerank_score"] = c.get("score", 0.0)
                c["rank"] = i
            return cands[:top_n]
        import numpy as np

        pairs = [(query, c["text"]) for c in cands]
        try:
            scores = model.predict(pairs, show_progress_bar=False)
            scores = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype="float32")))  # sigmoid -> [0,1]
        except Exception as exc:
            logger.warning("Rerank failed (%s); keeping retriever order.", exc)
            return cands[:top_n]
        for c, s in zip(cands, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(cands, key=lambda c: c["rerank_score"], reverse=True)[:top_n]
        for i, c in enumerate(ranked):
            c["rank"] = i
        return ranked


__all__ = ["Reranker"]
