"""Deterministic heuristics for the three agentic decision points.

These run with **no model and no API** so the agent is fully functional offline:

  1. ``analyze_query``      — route simple / multi-hop / unanswerable + rewrite.
  2. ``assess_sufficiency`` — is the reranked context enough? (CRAG verdict).
  3. ``assess_faithfulness``— is the answer grounded in the cited context?
     (the verified embedding/lexical-overlap fallback for the unverified NLI model).

The optional LLM orchestrator overrides 1–3 with stronger reasoning but always
falls back to these.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from ..config import AgentConfig

_STOP = set("the a an of to in on for and or is are was were be been being this that these those "
            "what which who whom whose when where why how did does do done has have had will would "
            "can could should may might must it its as at by with from into about".split())

_MULTIHOP_CUES = re.compile(r"\b(and|whose|that|which|before|after|compared to|as well as|also)\b", re.I)
_UNANSWERABLE_CUES = re.compile(r"\b(opinion|do you think|predict the future|will .* win)\b", re.I)

# light query-expansion synonyms (domain-agnostic)
_SYNONYMS = {
    "founder": ["co-founder", "created", "established", "started"],
    "university": ["college", "alma mater", "studied", "graduated"],
    "year": ["date", "when", "established", "founded"],
    "ceo": ["chief executive", "head", "leader"],
}


def content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2]


def analyze_query(question: str) -> Dict:
    """Decision point 1 — route + rewrite (rule-based)."""
    q = question.strip()
    is_multi = bool(_MULTIHOP_CUES.search(q)) and len(content_words(q)) >= 5
    qtype = "unanswerable" if _UNANSWERABLE_CUES.search(q) else ("multihop" if is_multi else "simple")
    sub_questions: List[str] = []
    if is_multi:
        # naive split on connectors into question-like fragments
        parts = re.split(r"\s*,?\s+\b(?:and|then)\b\s+", q, flags=re.I)
        sub_questions = [p.strip(" ?.") + "?" for p in parts if len(content_words(p)) >= 2]
        if len(sub_questions) < 2:
            sub_questions = []
            is_multi = False
            qtype = "simple"
    return {"qtype": qtype, "is_multi_hop": is_multi, "rewritten": q, "sub_questions": sub_questions}


def expand_query(query: str) -> str:
    """Add a few synonyms to widen retrieval on a failed attempt."""
    extra = []
    low = query.lower()
    for key, syns in _SYNONYMS.items():
        if key in low:
            extra.extend(syns)
    return query + (" " + " ".join(extra) if extra else "")


def assess_sufficiency(question: str, reranked: Sequence[Dict], cfg: AgentConfig) -> Dict:
    """Decision point 2 — SUFFICIENT / AMBIGUOUS / INSUFFICIENT (CRAG)."""
    if not reranked:
        return {"verdict": "INSUFFICIENT", "score": 0.0, "missing_terms": content_words(question)}
    top = reranked[0]
    score = float(top.get("rerank_score", top.get("score", 0.0)))
    # content-word coverage of the question by the top passages
    q_words = set(content_words(question))
    ctx = " ".join(c.get("text", "") for c in reranked[:3]).lower()
    covered = {w for w in q_words if w in ctx}
    missing = sorted(q_words - covered)
    coverage = len(covered) / max(1, len(q_words))

    if score >= cfg.tau_high and coverage >= 0.5:
        verdict = "SUFFICIENT"
    elif score < cfg.tau_low:
        verdict = "INSUFFICIENT"
    else:
        verdict = "AMBIGUOUS"
    return {"verdict": verdict, "score": round(score, 4), "coverage": round(coverage, 3), "missing_terms": missing[:8]}


def assess_faithfulness(answer: str, passages: Sequence[Dict], cfg: AgentConfig,
                        encoder=None) -> Dict:
    """Decision point 3 — is the answer grounded in the cited context?

    Verified fallback for the unverified NLI model: the fraction of the answer's
    content words that appear in the cited passages (lexical groundedness),
    optionally blended with embedding cosine if an encoder is supplied.
    """
    if not answer or not passages:
        return {"supported": False, "support_score": 0.0}
    ctx = " ".join(p.get("text", "") for p in passages).lower()
    a_words = content_words(answer)
    if not a_words:
        return {"supported": False, "support_score": 0.0}
    lexical = sum(1 for w in a_words if w in ctx) / len(a_words)

    score = lexical
    if encoder is not None:
        try:
            import numpy as np
            ans_emb = encoder.encode_queries([answer])[0]
            ctx_emb = encoder.encode_passages([" ".join(p.get("text", "") for p in passages)[:2000]])[0]
            cos = float(np.dot(ans_emb, ctx_emb))
            score = 0.5 * lexical + 0.5 * max(0.0, cos)
        except Exception:
            pass
    return {"supported": score >= cfg.faithfulness_threshold, "support_score": round(score, 4),
            "lexical_overlap": round(lexical, 4)}


def make_clarifying_question(question: str, missing_terms: Optional[List[str]] = None) -> str:
    if missing_terms:
        return (f"I couldn't find enough information about: {', '.join(missing_terms[:3])}. "
                "Could you clarify or rephrase your question?")
    return "I couldn't find a confident answer. Could you rephrase or add more detail to your question?"


__all__ = ["analyze_query", "expand_query", "assess_sufficiency", "assess_faithfulness",
           "make_clarifying_question", "content_words"]
