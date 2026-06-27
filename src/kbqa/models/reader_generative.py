"""Generative grounded reader (FLAN-T5).

Builds a strict, grounded prompt over the retrieved passages and generates an
abstractive answer that must come from the context — with an explicit
"say 'I don't know' if the answer is not in the context" instruction. Passages
are numbered ``[1] [2] ...`` so the answer can cite them; we attach the used
passages as citations. Optional upgrade over the extractive reader.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from ..config import GeneratorConfig
from ..logging_utils import get_logger
from .model_registry import has_model, resolve_latest
from .reader_extractive import ReadResult

logger = get_logger(__name__)

_IDK_RE = re.compile(r"\bi\s*(do\s*n['o]?t|don't)\s*know\b|\bnot\s+(in|found|available)\b", re.I)
_CITE_RE = re.compile(r"\[(\d+)\]")

PROMPT = (
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say \"I don't know.\"\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


class GenerativeReader:
    def __init__(self, cfg: Optional[GeneratorConfig] = None, model_dir: Optional[str] = None, device: Optional[str] = None):
        self.cfg = cfg or GeneratorConfig()
        self.device = device
        self._model = None
        self._tok = None
        self.name = None
        self._explicit_dir = model_dir

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            candidate = self.cfg.model_name
            if self._explicit_dir:
                latest = resolve_latest(self._explicit_dir)
                if has_model(latest):
                    candidate = str(latest)
            self._tok = AutoTokenizer.from_pretrained(candidate)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(candidate)
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(dev).eval()
            self.name = candidate

    def read(self, question: str, passages: Sequence[Dict], max_passages: int = 5) -> ReadResult:
        import torch

        cands = list(passages)[:max_passages]
        if not cands:
            return ReadResult()
        self._ensure()
        numbered = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(cands))
        prompt = PROMPT.format(context=numbered, question=question)
        enc = self._tok(prompt, return_tensors="pt", truncation=True, max_length=self.cfg.max_input_length)
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=self.cfg.max_target_length, num_beams=4)
        answer = self._tok.decode(out[0], skip_special_tokens=True).strip()

        if not answer or _IDK_RE.search(answer):
            return ReadResult(answer="I don't know.", abstain=True, per_passage=[])

        # citations: explicit [n] markers if present, else all passages used
        marked = [int(m) for m in _CITE_RE.findall(answer)]
        used = [cands[i - 1] for i in marked if 1 <= i <= len(cands)] or cands
        citations = [{"chunk_id": c.get("id"), "doc_id": c.get("doc_id"), "title": c.get("title"),
                      "quote": c["text"][:200]} for c in used]
        return ReadResult(answer=answer, score=1.0, abstain=False, citations=citations)


__all__ = ["GenerativeReader"]
