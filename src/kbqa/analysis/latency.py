"""Latency benchmark for the agentic RAG pipeline.

Measures wall-clock latency of two hot paths against the built-in sample KB so
the autopilot can report p50/p95/p99/mean without any dataset download or
trained checkpoint:

  * ``ask``    — the full agent loop ``RAGAgent.ask(question)`` (analyze →
                 retrieve → rerank → sufficiency → read → faithfulness gate).
  * ``search`` — the ``/search`` path only: ``retrieve_tool.run`` +
                 ``rerank_tool.run`` (what the API exposes for raw passage
                 lookup).

Questions are drawn from :data:`SAMPLE_QA` and cycled to ``n`` timed iterations
after ``warmup`` untimed iterations (so cold-start model loading / first-call
JIT does not skew the percentiles). Everything degrades gracefully: heavy deps
(torch, sentence_transformers, …) load lazily inside :class:`RAGAgent`, and any
failure during a single iteration is logged and skipped rather than crashing the
benchmark. Results are written under ``runs/benchmarks/benchmark-<stamp>.json``.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def _percentiles(samples_ms: List[float]) -> Dict[str, float]:
    """p50/p95/p99/mean (ms) for a list of latency samples.

    Uses ``numpy.percentile`` when available; falls back to a pure-Python
    nearest-rank implementation so a missing numpy never breaks the benchmark.
    Returns zeros for an empty sample set.
    """
    if not samples_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    try:
        import numpy as np

        arr = np.asarray(samples_ms, dtype=float)
        return {
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
        }
    except Exception as exc:  # pragma: no cover - numpy is a core dep, but stay safe
        logger.warning("numpy unavailable for percentiles (%s); using fallback.", exc)
        ordered = sorted(samples_ms)

        def _pct(p: float) -> float:
            # nearest-rank (linear-interpolation-free) percentile
            if len(ordered) == 1:
                return ordered[0]
            idx = (p / 100.0) * (len(ordered) - 1)
            lo = int(idx)
            hi = min(lo + 1, len(ordered) - 1)
            frac = idx - lo
            return ordered[lo] + frac * (ordered[hi] - ordered[lo])

        return {
            "p50_ms": round(_pct(50), 3),
            "p95_ms": round(_pct(95), 3),
            "p99_ms": round(_pct(99), 3),
            "mean_ms": round(sum(ordered) / len(ordered), 3),
        }


def _model_versions(agent) -> Dict[str, str]:
    """Best-effort snapshot of the component versions actually in play.

    Pulled from the tool ``.version`` attributes + orchestrator name so the
    benchmark JSON records exactly which retriever/reranker/reader produced the
    numbers (matters when comparing runs across checkpoints).
    """
    versions: Dict[str, str] = {}
    for attr in ("retrieve_tool", "rerank_tool", "generate_tool"):
        tool = getattr(agent, attr, None)
        if tool is not None and getattr(tool, "version", None):
            versions[getattr(tool, "name", attr)] = str(tool.version)
    orch = getattr(agent, "orchestrator", None)
    if orch is not None and getattr(orch, "name", None):
        versions["orchestrator"] = str(orch.name)
    return versions


def benchmark(cfg: AppConfig, n: int = 50, warmup: int = 5) -> dict:
    """Benchmark end-to-end ``ask`` and ``/search`` latency on the sample KB.

    Parameters
    ----------
    cfg:
        Application config; component models load lazily with graceful fallback.
    n:
        Number of *timed* iterations per path (questions cycled from SAMPLE_QA).
    warmup:
        Number of *untimed* warmup iterations to absorb cold-start cost.

    Returns
    -------
    dict
        ``{n, warmup, ask:{p50_ms,p95_ms,p99_ms,mean_ms},
        search:{...}, model_versions}`` — also written to
        ``runs/benchmarks/benchmark-<stamp>.json``.
    """
    from ..data.samples import SAMPLE_DOCS, SAMPLE_QA

    n = max(0, int(n))
    warmup = max(0, int(warmup))

    questions = [item["question"] for item in SAMPLE_QA if item.get("question")]
    if not questions:
        logger.warning("No sample questions available; benchmark is a no-op.")

    ask_ms: List[float] = []
    search_ms: List[float] = []
    model_versions: Dict[str, str] = {}

    # Build the agent on the built-in KB (no index download). If even this fails
    # we still emit a (zeroed) benchmark record so the autopilot never crashes.
    agent = None
    try:
        from ..agent.rag_agent import RAGAgent

        agent = RAGAgent(cfg, load_kb=False)
        model_versions = _model_versions(agent)
    except Exception as exc:
        logger.warning("Could not build benchmark agent (%s); returning empty timings.", exc)

    # Ingest the sample docs separately: if the encoder (sentence_transformers)
    # is missing this raises, but the agent is still usable (ask/search degrade
    # to abstain on an empty KB), so we keep timing the real call paths.
    if agent is not None:
        try:
            ingest_stats = agent.ingest(list(SAMPLE_DOCS))
            logger.info("Benchmark KB ingested: %s", ingest_stats)
        except Exception as exc:
            logger.warning("KB ingestion failed (%s); timing against empty KB.", exc)

    if agent is not None and questions:
        total = warmup + n
        rerank_top_n = getattr(cfg.reranker, "rerank_top_n", 5)
        retrieve_top_k = getattr(cfg.retriever, "top_k", 20)

        for i in range(total):
            question = questions[i % len(questions)]
            timed = i >= warmup  # first `warmup` iterations are untimed

            # ---- full agent path: ask() --------------------------------------
            try:
                t0 = time.perf_counter()
                agent.ask(question)
                dt = (time.perf_counter() - t0) * 1000.0
                if timed:
                    ask_ms.append(dt)
            except Exception as exc:
                logger.warning("ask() failed on iter %d (%r): %s", i, question[:60], exc)

            # ---- /search path: retrieve + rerank only ------------------------
            try:
                t0 = time.perf_counter()
                r_out = agent.retrieve_tool.run(query=question, top_k=retrieve_top_k)
                candidates = r_out.get("passages", []) if isinstance(r_out, dict) else []
                agent.rerank_tool.run(query=question, candidates=candidates, top_n=rerank_top_n)
                dt = (time.perf_counter() - t0) * 1000.0
                if timed:
                    search_ms.append(dt)
            except Exception as exc:
                logger.warning("search path failed on iter %d (%r): %s", i, question[:60], exc)

    results: Dict = {
        "n": n,
        "warmup": warmup,
        "ask": _percentiles(ask_ms),
        "search": _percentiles(search_ms),
        "model_versions": model_versions,
    }
    results["n_ask_samples"] = len(ask_ms)
    results["n_search_samples"] = len(search_ms)

    # Persist under runs/benchmarks/ for the autoreport. Never let a write error
    # take down the benchmark — return the in-memory result regardless.
    try:
        out_dir = run_dir() / "benchmarks"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"benchmark-{utc_stamp()}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info(
            "Benchmark -> %s | ask p50=%.1fms p95=%.1fms | search p50=%.1fms p95=%.1fms",
            out_path, results["ask"]["p50_ms"], results["ask"]["p95_ms"],
            results["search"]["p50_ms"], results["search"]["p95_ms"],
        )
    except Exception as exc:
        logger.warning("Could not write benchmark JSON (%s); returning in-memory result.", exc)

    return results


__all__ = ["benchmark"]
