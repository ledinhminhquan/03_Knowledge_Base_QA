"""Error analysis for the RAG QA agent.

Runs the full agent over the demo knowledge-base QA set and buckets every
prediction into an interpretable failure (or success) category:

    * ``correct``         — agent answered and the normalised answer matches gold
    * ``wrong_answer``    — agent answered but the normalised answer differs
    * ``missed_abstain``  — agent abstained on an *answerable* question (a miss)
    * ``correct_abstain`` — agent abstained on a truly *unanswerable* question
    * ``retrieval_miss``  — the gold answer string never appears in any retrieved
                            passage (the upstream retriever failed the reader)

The result is an aggregate report (per-category counts + up to ten example
*questions* per error category — never large passage dumps) written to
``runs/error_analysis/error-analysis-<stamp>.json`` and returned as a dict.

Everything degrades gracefully: dataset download failures fall back to the
built-in samples, and any heavy dependency is imported lazily so a missing
optional package can never crash the autopilot.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_now_iso, utc_stamp

logger = get_logger(__name__)

# Error buckets for which we collect example questions (successes are counted
# only — we surface *failures* for inspection).
_ERROR_CATEGORIES = ("wrong_answer", "missed_abstain", "retrieval_miss")
_ALL_CATEGORIES = ("correct", "wrong_answer", "missed_abstain",
                   "correct_abstain", "retrieval_miss")
_MAX_EXAMPLES = 10


def _load_qa_and_docs(cfg: AppConfig, limit: int) -> Tuple[List[Dict], List[Dict]]:
    """Return (docs, qa_pairs) for the demo KB, falling back to samples.

    ``qa_pairs`` are normalised to ``{question, answer(str|None)}`` regardless of
    source so downstream categorisation is uniform.
    """
    docs: List[Dict]
    qa_pairs: List[Dict]
    try:
        from ..data.dataset import load_demo_kb

        docs, raw_qa = load_demo_kb(cfg.data)
        qa_pairs = []
        for item in raw_qa:
            ans = item.get("answer")
            # Some datasets expose answers as a list/dict of spans.
            if isinstance(ans, (list, tuple)):
                ans = ans[0] if ans else None
            if isinstance(ans, dict):
                texts = ans.get("text")
                ans = (texts[0] if isinstance(texts, (list, tuple)) and texts else texts) or None
            qa_pairs.append({"question": item.get("question", ""), "answer": ans})
        if not docs or not qa_pairs:
            raise ValueError("empty demo KB")
    except Exception as exc:  # pragma: no cover - network/data dependent
        logger.warning("Demo KB unavailable (%s); using built-in SAMPLE_DOCS/SAMPLE_QA.", exc)
        from ..data.samples import SAMPLE_DOCS, SAMPLE_QA

        docs = list(SAMPLE_DOCS)
        qa_pairs = [{"question": q.get("question", ""), "answer": q.get("answer")}
                    for q in SAMPLE_QA]

    if limit and len(qa_pairs) > limit:
        qa_pairs = qa_pairs[:limit]
    return docs, qa_pairs


def _build_agent(cfg: AppConfig, docs: List[Dict]):
    """Build the demo KB and return a RAGAgent wired to that fresh index."""
    from ..agent.rag_agent import RAGAgent
    from ..data.corpus import build_kb_from_docs

    # Persist the index so a fresh agent loads it; also inject directly to be
    # robust to index-path / environment differences.
    retriever = build_kb_from_docs(cfg, docs)
    agent = RAGAgent(cfg, load_kb=False)
    agent.retriever = retriever
    agent.retrieve_tool.retriever = retriever
    return agent


def _answer_matches(pred: str, gold: str, normalize) -> bool:
    """SQuAD-style soft match: exact normalised equality or containment either way."""
    np = normalize(pred)
    ng = normalize(gold)
    if not ng:
        return False
    if np == ng:
        return True
    # Extractive readers often return a tighter/looser span than the gold string.
    return bool(np) and (ng in np or np in ng)


def _gold_in_retrieved(gold: str, state_dict: Dict[str, Any], normalize) -> bool:
    """True if the normalised gold answer appears in any retrieved/reranked passage."""
    ng = normalize(gold)
    if not ng:
        return False
    passages = (state_dict.get("retrieved") or []) + (state_dict.get("reranked") or [])
    for p in passages:
        text = p.get("text") if isinstance(p, dict) else None
        if text and ng in normalize(text):
            return True
    return False


def error_analysis(cfg: AppConfig, limit: int = 200) -> dict:
    """Categorise agent predictions over the demo KB and write an error report.

    Parameters
    ----------
    cfg:
        Loaded application configuration.
    limit:
        Maximum number of QA pairs to evaluate (caps runtime).

    Returns
    -------
    dict
        ``{n_questions, limit, counts:{...}, rates:{...}, examples:{...},
        artifact, generated_at, ...}`` — also persisted under
        ``runs/error_analysis/error-analysis-<stamp>.json``.
    """
    from ..data.preprocessing import normalize_answer

    report: Dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "limit": limit,
        "categories": list(_ALL_CATEGORIES),
        "counts": {c: 0 for c in _ALL_CATEGORIES},
        "examples": {c: [] for c in _ERROR_CATEGORIES},
        "n_questions": 0,
        "n_errors": 0,
        "n_skipped": 0,
    }

    docs, qa_pairs = _load_qa_and_docs(cfg, limit)
    report["n_questions"] = len(qa_pairs)
    report["dataset_docs"] = len(docs)

    try:
        agent = _build_agent(cfg, docs)
    except Exception as exc:
        logger.warning("Could not build agent/KB for error analysis (%s); returning partial.", exc)
        report["error"] = f"agent_unavailable: {exc}"
        return _write(report)

    counts = report["counts"]
    examples = report["examples"]

    for i, qa in enumerate(qa_pairs):
        question = (qa.get("question") or "").strip()
        gold = qa.get("answer")
        if not question:
            report["n_skipped"] += 1
            continue

        try:
            state = agent.ask(question, query_id=f"ea-{i}")
            sd = state.to_dict()
        except Exception as exc:  # never let one bad query abort the whole run
            logger.warning("ask() failed for q%d (%s); skipping.", i, exc)
            report["n_skipped"] += 1
            continue

        status = sd.get("status")
        answered = status == "answered"
        # A None/empty gold answer marks a genuinely unanswerable question.
        gold_str = gold if isinstance(gold, str) else ""
        answerable = bool(gold_str.strip())

        if answered:
            pred = sd.get("answer") or ""
            if answerable and _answer_matches(pred, gold_str, normalize_answer):
                category = "correct"
            else:
                # Distinguish a reader mistake from an upstream retrieval failure:
                # if the gold string was never retrieved, it's a retrieval miss.
                if answerable and not _gold_in_retrieved(gold_str, sd, normalize_answer):
                    category = "retrieval_miss"
                else:
                    category = "wrong_answer"
        else:
            # Agent abstained / declined (insufficient | no_answer | needs_clarification).
            if answerable:
                category = "missed_abstain"
            else:
                category = "correct_abstain"

        counts[category] += 1
        if category in examples and len(examples[category]) < _MAX_EXAMPLES:
            ex: Dict[str, Any] = {"question": question, "status": status}
            if answerable:
                ex["gold"] = gold_str
                ex["predicted"] = (sd.get("answer") or "")[:160]
            examples[category].append(ex)

    n = report["n_questions"] - report["n_skipped"]
    report["n_evaluated"] = n
    report["n_errors"] = sum(counts[c] for c in _ERROR_CATEGORIES)
    report["rates"] = {c: round(counts[c] / n, 4) if n else 0.0 for c in _ALL_CATEGORIES}
    report["accuracy"] = round((counts["correct"] + counts["correct_abstain"]) / n, 4) if n else 0.0

    return _write(report)


def _write(report: Dict[str, Any]) -> Dict[str, Any]:
    """Persist the report under runs/error_analysis/ and stamp the artifact path."""
    try:
        out_dir = run_dir() / "error_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"error-analysis-{utc_stamp()}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["artifact"] = str(out_path)
        logger.info("Wrote error analysis: %s", out_path)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        logger.warning("Could not write error-analysis artifact (%s).", exc)
        report.setdefault("artifact", None)
    return report


__all__ = ["error_analysis"]
