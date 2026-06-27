"""Model-backed agent tools (retrieve / rerank / generate).

Each tool has a ``name`` + ``version`` and a ``run(**kwargs)->dict`` that times
itself and never raises past the orchestrator (errors become ``{ok: False,
error}``). The query-analysis / sufficiency / faithfulness decision points are
handled by :mod:`kbqa.agent.policy` (rule) or the LLM orchestrator and traced by
the agent directly.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from ..logging_utils import get_logger

logger = get_logger(__name__)


class Tool:
    name: str = "tool"
    version: str = "1.0.0"

    def _run(self, **kwargs) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def run(self, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            out = self._run(**kwargs)
            out.setdefault("ok", True)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", self.name, exc)
            out = {"ok": False, "error": str(exc)}
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out


class RetrieveTool(Tool):
    name = "retrieve"

    def __init__(self, retriever=None):
        self.retriever = retriever
        self.version = f"retriever-{getattr(getattr(retriever, 'encoder', None), 'name', None) or '1.0.0'}"

    def _run(self, query: str = "", top_k: int = 20, **_) -> Dict[str, Any]:
        if self.retriever is None:
            return {"passages": []}
        return {"passages": self.retriever.retrieve(query, top_k=top_k)}


class RerankTool(Tool):
    name = "rerank"

    def __init__(self, reranker=None):
        self.reranker = reranker
        self.version = f"reranker-{getattr(reranker, 'name', None) or '1.0.0'}"

    def _run(self, query: str = "", candidates: Optional[Sequence[Dict]] = None, top_n: int = 5, **_) -> Dict[str, Any]:
        cands = list(candidates or [])
        if self.reranker is None or not cands:
            # no reranker: keep retriever order but expose a rerank_score for consistency
            kept = cands[:top_n]
            for i, c in enumerate(kept):
                c.setdefault("rerank_score", c.get("score", 0.0))
                c["rank"] = i
            return {"passages": kept}
        return {"passages": self.reranker.rerank(query, cands, top_n=top_n)}


class GenerateTool(Tool):
    name = "generate"

    def __init__(self, reader=None, mode: str = "extractive"):
        self.reader = reader
        self.mode = mode
        self.version = f"{mode}-{getattr(reader, 'name', None) or '1.0.0'}"

    def _run(self, question: str = "", passages: Optional[Sequence[Dict]] = None, **_) -> Dict[str, Any]:
        ps = list(passages or [])
        if self.reader is None or not ps:
            return {"answer": "", "abstain": True, "citations": [], "score": 0.0}
        res = self.reader.read(question, ps)
        return res.to_dict()


__all__ = ["Tool", "RetrieveTool", "RerankTool", "GenerateTool"]
