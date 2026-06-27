"""End-to-end RAG evaluation: retrieval recall + answer EM/F1 + abstention.

Builds the demo KB, then measures:
  * **Retrieval** — Recall@k proxy (does a retrieved passage contain the gold
    answer string?) for BM25-only vs the hybrid retriever.
  * **Answer quality** — EM / F1 of the agent's final answer vs the gold answer.
  * **Abstention** — fraction of unanswerable questions correctly abstained.

Writes a JSON snapshot under ``runs/eval-<stamp>/`` for the autoreport.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..data.preprocessing import normalize_answer
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def _f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def _answer_str(a) -> str:
    if isinstance(a, dict):
        a = a.get("text") or a.get("answer") or ""
    if isinstance(a, list):
        a = a[0] if a else ""
    return str(a or "")


def evaluate(cfg: AppConfig, limit: Optional[int] = None, k: int = 5) -> Dict:
    from ..agent.rag_agent import RAGAgent
    from ..data.corpus import build_kb_from_docs
    from ..data.dataset import load_demo_kb
    from ..data.samples import SAMPLE_DOCS, SAMPLE_QA

    # Load demo KB + QA (fallback to built-in samples if dataset unavailable)
    try:
        docs, qa = load_demo_kb(cfg.data, limit_corpus=limit)
    except Exception as exc:
        logger.warning("Demo KB unavailable (%s); using built-in samples.", exc)
        docs = list(SAMPLE_DOCS)
        qa = [{"question": x["question"], "answer": x["answer"]} for x in SAMPLE_QA]

    n_q = min(len(qa), limit or 100)
    qa = qa[:n_q]

    agent = RAGAgent(cfg, load_kb=False)
    agent.retriever = build_kb_from_docs(cfg, docs, save=False)
    agent.retrieve_tool.retriever = agent.retriever

    bm25_hits = hybrid_hits = 0
    em = f1 = abstain_correct = abstain_total = 0
    answered = 0
    for item in qa:
        q = item["question"]
        gold = _answer_str(item.get("answer"))
        gold_norm = normalize_answer(gold)
        # retrieval recall proxy
        hybrid = agent.retriever.retrieve(q, top_k=k)
        if gold_norm and any(gold_norm in normalize_answer(p["text"]) for p in hybrid):
            hybrid_hits += 1
        if agent.retriever.bm25 is not None:
            bm = agent.retriever.bm25.search(q, top_k=k)
            if gold_norm and any(gold_norm in normalize_answer(p.text) for p, _ in bm):
                bm25_hits += 1
        # end-to-end answer
        state = agent.ask(q)
        is_unanswerable = not gold
        if is_unanswerable:
            abstain_total += 1
            if state.status and state.status.value != "answered":
                abstain_correct += 1
        else:
            answered += 1
            pred = state.answer if state.status and state.status.value == "answered" else ""
            em += float(normalize_answer(pred) == gold_norm)
            f1 += _f1(pred, gold)

    results = {
        "n_questions": len(qa), "k": k,
        "retrieval": {
            "recall@{}_hybrid".format(k): round(hybrid_hits / max(1, len(qa)), 4),
            "recall@{}_bm25".format(k): round(bm25_hits / max(1, len(qa)), 4),
        },
        "answer": {
            "exact_match": round(100 * em / max(1, answered), 2),
            "f1": round(100 * f1 / max(1, answered), 2),
            "n_answerable": answered,
        },
        "abstention": {
            "abstain_recall": round(abstain_correct / max(1, abstain_total), 4),
            "n_unanswerable": abstain_total,
        },
    }
    results["summary"] = {
        "recall@{}_hybrid".format(k): results["retrieval"]["recall@{}_hybrid".format(k)],
        "answer_f1": results["answer"]["f1"],
        "abstain_recall": results["abstention"]["abstain_recall"],
    }
    out = run_dir() / f"eval-{utc_stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Eval -> %s | %s", out / "eval.json", results["summary"])
    return results


__all__ = ["evaluate"]
