"""BM25 sparse retriever — the mandatory baseline + lexical fallback.

BM25 is the classic strong sparse baseline that the dense retriever must beat to
justify its cost. It also serves as the **zero-dependency fallback** retriever
when neither FAISS nor sentence-transformers is available, so the RAG pipeline
always returns passages.

Uses ``rank_bm25`` if installed; otherwise falls back to a NumPy TF-IDF cosine
ranker so the system still runs.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from ..index.vector_store import Passage
from ..logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tok(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self):
        self.passages: List[Passage] = []
        self._bm25 = None
        self._tfidf = None        # (vectorizer, matrix) fallback
        self._backend = "none"

    def index(self, passages: Sequence[Passage]) -> "BM25Retriever":
        self.passages = list(passages)
        corpus = [p.text for p in self.passages]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([_tok(t) for t in corpus])
            self._backend = "bm25"
        except Exception as exc:
            logger.info("rank_bm25 unavailable (%s); using TF-IDF cosine fallback.", exc)
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            mat = vec.fit_transform(corpus)
            self._tfidf = (vec, mat)
            self._backend = "tfidf"
        logger.info("BM25 baseline indexed %d passages (backend=%s)", len(self.passages), self._backend)
        return self

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Passage, float]]:
        if not self.passages:
            return []
        top_k = min(top_k, len(self.passages))
        if self._backend == "bm25":
            import numpy as np
            scores = self._bm25.get_scores(_tok(query))
            order = np.argsort(scores)[::-1][:top_k]
            return [(self.passages[int(i)], float(scores[int(i)])) for i in order]
        if self._backend == "tfidf":
            import numpy as np
            vec, mat = self._tfidf
            q = vec.transform([query])
            sims = (mat @ q.T).toarray().ravel()
            order = np.argsort(sims)[::-1][:top_k]
            return [(self.passages[int(i)], float(sims[int(i)])) for i in order]
        return []


__all__ = ["BM25Retriever"]
