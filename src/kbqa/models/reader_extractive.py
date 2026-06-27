"""Extractive span reader with native abstention (SQuAD2 null score).

``deepset/roberta-base-squad2`` is trained on SQuAD 2.0, so it can return an
empty answer ("no answer") when the passage doesn't contain one — this is the
"I don't know" mechanism that keeps the RAG system honest. We run the reader over
each top passage, pick the highest-scoring non-null span, and attach the source
passage as the **citation**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..config import ReaderConfig
from ..logging_utils import get_logger
from .model_registry import has_model, resolve_latest

logger = get_logger(__name__)


@dataclass
class ReadResult:
    answer: str = ""
    score: float = 0.0
    abstain: bool = True
    citations: List[Dict] = field(default_factory=list)
    per_passage: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"answer": self.answer, "score": round(self.score, 4), "abstain": self.abstain,
                "citations": self.citations}


class ExtractiveReader:
    def __init__(self, cfg: Optional[ReaderConfig] = None, model_dir: Optional[str] = None, device: Optional[str] = None):
        self.cfg = cfg or ReaderConfig()
        self.device = device
        self._pipe = None
        self.name = None
        self._explicit_dir = model_dir

    @property
    def pipe(self):
        if self._pipe is None:
            from transformers import pipeline

            candidate = self.cfg.model_name
            if self._explicit_dir:
                latest = resolve_latest(self._explicit_dir)
                if has_model(latest):
                    candidate = str(latest)
            device_id = 0 if (self.device == "cuda") else -1
            self._pipe = pipeline("question-answering", model=candidate, tokenizer=candidate,
                                  framework="pt", device=device_id)
            self.name = candidate
        return self._pipe

    def read(self, question: str, passages: Sequence[Dict], top_passages: int = 5) -> ReadResult:
        cands = list(passages)[:top_passages]
        if not cands:
            return ReadResult()
        pipe = self.pipe
        best = None
        per_passage = []
        for c in cands:
            try:
                res = pipe(question=question, context=c["text"], handle_impossible_answer=True,
                           max_seq_len=self.cfg.max_length, doc_stride=self.cfg.doc_stride,
                           max_answer_len=self.cfg.max_answer_length)
            except Exception as exc:
                logger.debug("reader error on passage %s: %s", c.get("id"), exc)
                continue
            ans = (res.get("answer") or "").strip()
            score = float(res.get("score", 0.0))
            per_passage.append({"id": c.get("id"), "answer": ans, "score": score})
            if ans and (best is None or score > best[1]):
                best = (ans, score, c)

        if best is None:
            return ReadResult(per_passage=per_passage)
        ans, score, passage = best
        abstain = score < self.cfg.null_score_threshold or not ans
        citation = {"chunk_id": passage.get("id"), "doc_id": passage.get("doc_id"),
                    "title": passage.get("title"), "quote": ans, "score": round(score, 4)}
        return ReadResult(answer=ans, score=score, abstain=abstain,
                          citations=[] if abstain else [citation], per_passage=per_passage)


__all__ = ["ExtractiveReader", "ReadResult"]
