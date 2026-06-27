"""RAG groundedness / faithfulness audit.

This module answers the question every RAG operator must answer before trusting a
deployed system: *when the agent gives an answer, is that answer actually
supported by the passages it retrieved, or is it hallucinating?*

We run the live :class:`~kbqa.agent.rag_agent.RAGAgent` over the demo QA set and,
for every **answered** question, recompute groundedness with
:func:`kbqa.agent.policy.assess_faithfulness` against the reranked context the
agent actually used. From the per-question verdicts we aggregate a small set of
trust metrics:

  * ``mean_support_score``        — average groundedness over answered questions.
  * ``grounded_rate``             — fraction of answers the policy marks supported.
  * ``answered_with_citations``   — fraction of answers that carry >=1 citation.
  * ``hallucination_rate``        — answered but support_score < threshold.
  * ``abstain_rate``              — fraction of questions the agent declined.

Everything imports the existing public API and degrades gracefully: a missing
optional dependency (torch / sentence-transformers / a trained model) downgrades
the encoder-blended score to the lexical-overlap fallback rather than crashing
the autopilot. We aggregate stats only and keep a few illustrative per-question
rows (public sample questions) for the report.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_now_iso, utc_stamp

logger = get_logger(__name__)


def _passage_dicts(state) -> List[Dict[str, Any]]:
    """Best-effort extraction of the reranked passages (as ``{'text': ...}`` dicts).

    ``state.reranked`` holds :class:`RetrievedPassage` dataclasses; the policy
    expects plain dicts with a ``text`` key. Convert defensively so a different
    shape (already-dicts, missing ``to_dict``) never raises.
    """
    out: List[Dict[str, Any]] = []
    for p in (getattr(state, "reranked", None) or []):
        if isinstance(p, dict):
            out.append(p)
        elif hasattr(p, "to_dict"):
            try:
                out.append(p.to_dict())
            except Exception:
                out.append({"text": getattr(p, "text", ""), "id": getattr(p, "id", "")})
        else:
            out.append({"text": getattr(p, "text", ""), "id": getattr(p, "id", "")})
    return out


def _rerun_context(agent, question: str) -> List[Dict[str, Any]]:
    """Fallback: re-run retrieve + rerank to recover passages if state had none.

    Mirrors the agent's own retrieve→rerank step using its public tools so the
    audit still has context to score against even when ``state.reranked`` is
    empty (e.g. the agent abstained before populating it).
    """
    try:
        cfg = agent.cfg
        r_out = agent.retrieve_tool.run(query=question, top_k=cfg.retriever.top_k)
        cands = r_out.get("passages", [])
        rr_out = agent.rerank_tool.run(query=question, candidates=cands,
                                       top_n=cfg.reranker.rerank_top_n)
        return rr_out.get("passages", []) or cands
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Faithfulness audit: re-run of retrieve/rerank failed (%s).", exc)
        return []


def _build_agent_with_kb(cfg: AppConfig):
    """Construct a RAGAgent over the demo KB (dataset → fallback SAMPLE_DOCS).

    Returns ``(agent, qa_pairs)``. Never raises on a missing dataset/model: it
    degrades to the built-in sample corpus + QA so the audit always runs.
    """
    from ..agent.rag_agent import RAGAgent
    from ..data.corpus import build_kb_from_docs
    from ..data.samples import SAMPLE_DOCS, SAMPLE_QA

    # Prefer the richer demo KB dataset; fall back to the offline samples.
    try:
        from ..data.dataset import load_demo_kb
        docs, qa = load_demo_kb(cfg.data, limit_corpus=None)
    except Exception as exc:
        logger.warning("Demo KB dataset unavailable (%s); using built-in samples.", exc)
        docs = list(SAMPLE_DOCS)
        qa = [{"question": x["question"], "answer": x["answer"]} for x in SAMPLE_QA]

    agent = RAGAgent(cfg, load_kb=False)
    try:
        agent.retriever = build_kb_from_docs(cfg, docs, save=False)
        agent.retrieve_tool.retriever = agent.retriever
    except Exception as exc:
        logger.warning("Faithfulness audit: KB build failed (%s); using built-in samples.", exc)
        docs = list(SAMPLE_DOCS)
        agent.retriever = build_kb_from_docs(cfg, docs, save=False)
        agent.retrieve_tool.retriever = agent.retriever
        qa = [{"question": x["question"], "answer": x["answer"]} for x in SAMPLE_QA]
    return agent, qa


def faithfulness_eval(cfg: AppConfig, limit: int = 100) -> dict:
    """Audit how well the agent's answers are grounded in their retrieved context.

    Parameters
    ----------
    cfg:
        The application configuration (thresholds live in ``cfg.agent``).
    limit:
        Cap on the number of QA pairs to evaluate.

    Returns
    -------
    dict
        Aggregate groundedness metrics, an illustrative summary, and a handful of
        per-question rows. Also written to ``runs/faithfulness/faithfulness-<stamp>.json``.
    """
    from ..agent import policy

    threshold = float(cfg.agent.faithfulness_threshold)

    try:
        agent, qa = _build_agent_with_kb(cfg)
    except Exception as exc:  # pragma: no cover - last-resort guard
        logger.warning("Faithfulness audit could not build the agent (%s).", exc)
        result = {
            "error": f"agent_build_failed: {exc}",
            "n_questions": 0,
            "faithfulness_threshold": threshold,
        }
        return _write(result)

    n_q = min(len(qa), max(0, int(limit)) or 0) if limit else len(qa)
    qa = qa[:n_q]

    encoder = getattr(agent.retriever, "encoder", None)

    # --- per-question audit ---------------------------------------------------
    n_answered = 0
    n_grounded = 0
    n_with_citations = 0
    n_hallucinated = 0
    n_abstained = 0
    support_sum = 0.0
    lexical_sum = 0.0
    examples: List[Dict[str, Any]] = []

    for item in qa:
        question = item.get("question", "")
        if not question:
            continue
        try:
            state = agent.ask(question)
        except Exception as exc:  # pragma: no cover - one bad question never aborts the audit
            logger.warning("Faithfulness audit: ask() failed for %r (%s).", question[:60], exc)
            continue

        status = state.status.value if state.status is not None else None
        if status != "answered":
            n_abstained += 1
            continue

        answer = (state.answer or "").strip()
        passages = _passage_dicts(state) or _rerun_context(agent, question)

        try:
            faith = policy.assess_faithfulness(answer, passages, cfg.agent, encoder=encoder)
        except Exception as exc:  # pragma: no cover - encoder hiccup → lexical-only retry
            logger.warning("assess_faithfulness failed (%s); retrying lexical-only.", exc)
            try:
                faith = policy.assess_faithfulness(answer, passages, cfg.agent, encoder=None)
            except Exception:
                faith = {"supported": False, "support_score": 0.0, "lexical_overlap": 0.0}

        support = float(faith.get("support_score", 0.0))
        lexical = float(faith.get("lexical_overlap", support))
        supported = bool(faith.get("supported", support >= threshold))
        has_citations = bool(state.citations)

        n_answered += 1
        support_sum += support
        lexical_sum += lexical
        if supported:
            n_grounded += 1
        if has_citations:
            n_with_citations += 1
        if support < threshold:
            n_hallucinated += 1  # answered but not adequately grounded

        if len(examples) < 8:
            examples.append({
                "question": question,
                "answer": answer[:200],
                "support_score": round(support, 4),
                "lexical_overlap": round(lexical, 4),
                "supported": supported,
                "n_citations": len(state.citations),
                "n_passages": len(passages),
            })

    # --- aggregate ------------------------------------------------------------
    denom = max(1, n_answered)
    n_total = n_answered + n_abstained
    mean_support = round(support_sum / denom, 4)
    grounded_rate = round(n_grounded / denom, 4)
    cited_rate = round(n_with_citations / denom, 4)
    hallucination_rate = round(n_hallucinated / denom, 4)
    abstain_rate = round(n_abstained / max(1, n_total), 4)

    result: Dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "n_questions": len(qa),
        "faithfulness_threshold": threshold,
        "encoder_available": encoder is not None,
        "counts": {
            "answered": n_answered,
            "abstained": n_abstained,
            "grounded": n_grounded,
            "answered_with_citations": n_with_citations,
            "hallucinated": n_hallucinated,
        },
        "metrics": {
            "mean_support_score": mean_support,
            "mean_lexical_overlap": round(lexical_sum / denom, 4),
            "grounded_rate": grounded_rate,
            "answered_with_citations_rate": cited_rate,
            "hallucination_rate": hallucination_rate,
            "abstain_rate": abstain_rate,
        },
        "examples": examples,
    }
    result["summary"] = {
        "mean_support_score": mean_support,
        "grounded_rate": grounded_rate,
        "hallucination_rate": hallucination_rate,
        "abstain_rate": abstain_rate,
        "verdict": _verdict(mean_support, hallucination_rate, threshold),
    }

    return _write(result)


def _verdict(mean_support: float, hallucination_rate: float, threshold: float) -> str:
    """One-line illustrative read on the audit for the report."""
    if mean_support >= max(0.6, threshold + 0.2) and hallucination_rate <= 0.1:
        return ("Answers are well grounded: mean support comfortably clears the "
                f"{threshold:.2f} threshold and hallucinations are rare.")
    if hallucination_rate >= 0.3:
        return ("Elevated hallucination rate — a notable share of answers fall below "
                "the groundedness threshold; tighten the faithfulness gate or retrieval.")
    return ("Groundedness is moderate; most answers are supported but the margin over "
            "the threshold is slim — monitor and consider a stricter citation gate.")


def _write(result: Dict[str, Any]) -> Dict[str, Any]:
    """Persist the audit JSON under ``runs/faithfulness/`` and return it."""
    try:
        out_dir = run_dir() / "faithfulness"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"faithfulness-{utc_stamp()}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["artifact_path"] = str(out_path)
        logger.info("Faithfulness audit -> %s | %s", out_path, result.get("summary", {}))
    except Exception as exc:  # pragma: no cover - never fail the autopilot on IO
        logger.warning("Could not write faithfulness audit JSON (%s).", exc)
    return result


__all__ = ["faithfulness_eval"]
