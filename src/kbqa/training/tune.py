"""Lightweight hyperparameter tuning for the reader's abstention threshold.

The most impactful, cheap knob in a SQuAD2 RAG reader is the **null-score
threshold** that trades answer recall against abstention precision. We sweep it
on the demo QA and pick the value that maximises a blended EM + abstain-recall
score (no GPU training needed).
"""

from __future__ import annotations

import json
from typing import Dict

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def tune_reader(cfg: AppConfig) -> Dict:
    from ..agent.rag_agent import RAGAgent
    from ..data.corpus import build_kb_from_docs
    from ..data.samples import SAMPLE_DOCS, SAMPLE_QA
    from ..data.preprocessing import normalize_answer

    docs = list(SAMPLE_DOCS)
    qa = SAMPLE_QA
    best = {"threshold": cfg.reader.null_score_threshold, "score": -1.0}
    sweep = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
    results = []
    for thr in sweep:
        c = cfg
        c.reader.null_score_threshold = thr
        agent = RAGAgent(c, load_kb=False)
        agent.retriever = build_kb_from_docs(c, docs, save=False)
        agent.retrieve_tool.retriever = agent.retriever
        em = ab = ab_total = answered = 0
        for item in qa:
            state = agent.ask(item["question"])
            gold = item.get("answer")
            if not gold:
                ab_total += 1
                ab += int(state.status and state.status.value != "answered")
            else:
                answered += 1
                pred = state.answer if state.status and state.status.value == "answered" else ""
                em += int(normalize_answer(pred) == normalize_answer(str(gold)))
        score = (em / max(1, answered)) * 0.6 + (ab / max(1, ab_total)) * 0.4
        results.append({"threshold": thr, "em": em, "abstain": ab, "score": round(score, 3)})
        if score > best["score"]:
            best = {"threshold": thr, "score": round(score, 3)}
    out = run_dir() / f"tune-reader-{utc_stamp()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "sweep": results}, indent=2), encoding="utf-8")
    logger.info("Best null_score_threshold=%s (score=%s)", best["threshold"], best["score"])
    return {"best": best, "sweep": results}


__all__ = ["tune_reader"]
