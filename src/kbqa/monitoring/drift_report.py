"""Production monitoring + drift detection for the KBQA serving layer.

This module turns the append-only query log (one JSON object per ``/ask`` call,
written by :class:`kbqa.agent.rag_agent.RAGAgent` via
:class:`kbqa.logging_utils.JsonlLogger`) into the two reports an on-call owner
actually reads:

* :func:`monitoring_report` — a *health snapshot* of a single log: how many
  queries, how the answer-status distribution splits, the abstention rate, and
  the central tendency / tails of confidence & faithfulness. These are the
  signals that tell you whether the agent is quietly degrading (e.g. abstaining
  far more often than at launch, or answering with collapsing faithfulness).

* :func:`drift_report` — a *comparison* of two logs (a trusted reference window
  vs. a current window) using the Population Stability Index (PSI) on the
  answer-status distribution. PSI is the standard, threshold-friendly drift
  metric in production ML monitoring: ``PSI > 0.2`` conventionally flags a
  meaningful population shift worth a human looking at.

Design notes
------------
* The query-log schema we read is exactly what ``RAGAgent._finish`` writes::

      {"ts": "<iso8601>", "event": "ask", "query_id": "...",
       "status": "answered|insufficient|no_answer|needs_clarification",
       "confidence": <float>, "faithfulness": <float>,
       "n_citations": <int>, "multihop": <bool>}

  We stay defensive about every field — real logs rot, get truncated mid-write,
  or pre-date a schema change — so a malformed line is skipped, not fatal.
* ``numpy`` is imported lazily inside the PSI helper and degrades gracefully:
  if it is missing we fall back to a pure-Python PSI so the autopilot never
  crashes on an optional dependency.
* Empty / missing logs return ``{'note': 'no logs'}`` rather than raising, so a
  fresh deployment with zero traffic reports cleanly.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)

# Canonical answer statuses (mirrors agent.state.AnswerStatus values). We pin the
# order so PSI buckets line up across two independent logs even when one of them
# never produced a given status.
_STATUSES = ("answered", "insufficient", "no_answer", "needs_clarification")

# Statuses that represent the agent declining to give a grounded answer.
_ABSTAIN_STATUSES = ("insufficient", "no_answer", "needs_clarification")

# Conventional PSI alarm thresholds used across the industry:
#   PSI < 0.1  → no significant shift
#   0.1–0.2    → moderate shift, keep an eye on it
#   PSI > 0.2  → significant shift, investigate
_PSI_DRIFT_THRESHOLD = 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Log loading
# ─────────────────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL query log, tolerating blank / malformed / partial lines.

    A monitoring reader that crashes on the first bad line is worse than useless,
    so each line is parsed independently and bad lines are counted-and-skipped.
    """
    records: List[Dict[str, Any]] = []
    bad = 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    bad += 1
    except FileNotFoundError:
        return []
    except Exception as exc:  # pragma: no cover - unexpected IO error
        logger.warning("Could not read query log %s: %s", path, exc)
        return []
    if bad:
        logger.warning("Skipped %d malformed line(s) in %s", bad, path)
    return records


def _ask_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the ``ask`` query events (ignore any other JsonlLogger events)."""
    out = []
    for r in records:
        ev = r.get("event")
        # Pre-schema or hand-written logs may omit ``event``; accept anything
        # that at least carries a recognisable status field.
        if ev == "ask" or ("status" in r and ev is None):
            out.append(r)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Small statistics helpers (numpy-free, so they always work)
# ─────────────────────────────────────────────────────────────────────────────

def _floats(records: List[Dict[str, Any]], key: str) -> List[float]:
    """Pull a numeric field from records, dropping missing / non-numeric values."""
    vals: List[float] = []
    for r in records:
        v = r.get(key)
        if v is None or isinstance(v, bool):
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return vals


def _mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _percentile(xs: List[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile (matches numpy's default 'linear' method)."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _summ(records: List[Dict[str, Any]], key: str) -> Dict[str, Optional[float]]:
    """mean / p95 / min / max summary for a numeric field, rounded for reports."""
    xs = _floats(records, key)
    if not xs:
        return {"mean": None, "p95": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(_mean(xs), 4),
        "p95": round(_percentile(xs, 95.0), 4),
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
        "n": len(xs),
    }


def _status_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count answer statuses, bucketing anything unexpected under 'other'."""
    counts = {s: 0 for s in _STATUSES}
    other = 0
    for r in records:
        s = r.get("status")
        if s in counts:
            counts[s] += 1
        else:
            other += 1
    if other:
        counts["other"] = other
    return counts


def _time_range(records: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """First / last timestamps in the log (ISO-8601 strings sort chronologically)."""
    ts = [r.get("ts") for r in records if isinstance(r.get("ts"), str)]
    if not ts:
        return {"start": None, "end": None}
    ts.sort()
    return {"start": ts[0], "end": ts[-1]}


# ─────────────────────────────────────────────────────────────────────────────
# PSI (Population Stability Index)
# ─────────────────────────────────────────────────────────────────────────────

def _distribution(counts: Dict[str, int]) -> List[float]:
    """Normalise the canonical status buckets into a probability vector.

    We deliberately use the fixed ``_STATUSES`` ordering (ignoring any 'other'
    bucket) so reference and current logs are always compared bucket-for-bucket.
    """
    total = sum(counts.get(s, 0) for s in _STATUSES)
    if total <= 0:
        return [0.0 for _ in _STATUSES]
    return [counts.get(s, 0) / total for s in _STATUSES]


def _psi(reference: List[float], current: List[float], eps: float = 1e-6) -> float:
    """Population Stability Index between two probability vectors.

        PSI = Σ (cur_i - ref_i) * ln(cur_i / ref_i)

    Bins with zero mass are floored to ``eps`` to keep the log finite — the
    standard practical fix for the empty-bucket case. Computed with numpy when
    available, otherwise pure Python (so a missing optional dep never crashes).
    """
    try:
        import numpy as np  # lazy: optional heavy dep

        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)
        ref = np.clip(ref, eps, None)
        cur = np.clip(cur, eps, None)
        return float(np.sum((cur - ref) * np.log(cur / ref)))
    except Exception:
        # Pure-Python fallback — identical formula, no numpy required.
        from math import log

        total = 0.0
        for r, c in zip(reference, current):
            r = max(r, eps)
            c = max(c, eps)
            total += (c - r) * log(c / r)
        return float(total)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def monitoring_report(cfg: AppConfig, log_path: str = None) -> dict:
    """Aggregate the serving query log into a single health snapshot.

    Reads the JSONL query log at ``log_path`` (or ``cfg.serving.query_log_path``
    when omitted) and computes:

    * ``total`` queries and the per-status counts + rates,
    * ``abstain_rate`` (share of insufficient / no_answer / needs_clarification),
    * ``multihop_rate``,
    * mean / p95 (+ min/max) of ``confidence`` and ``faithfulness``,
    * the ``time_range`` covered by the log.

    The report is also persisted under ``runs/monitoring/monitoring-<stamp>.json``
    for the autoreport pipeline. Missing or empty logs return
    ``{'note': 'no logs'}`` (still written to disk for traceability).

    Parameters
    ----------
    cfg:
        Application config; supplies the default log path and is echoed for
        provenance.
    log_path:
        Optional explicit path to a JSONL query log. Falls back to
        ``cfg.serving.query_log_path``.

    Returns
    -------
    dict
        The aggregated report (also written to ``runs/monitoring/``).
    """
    path = Path(log_path) if log_path else Path(cfg.serving.query_log_path)
    records = _ask_records(_read_jsonl(path))
    total = len(records)

    if total == 0:
        report: Dict[str, Any] = {
            "note": "no logs",
            "log_path": str(path),
            "generated_at": utc_stamp(),
        }
        _write_report(report)
        return report

    counts = _status_counts(records)
    rates = {s: round(c / total, 4) for s, c in counts.items()}

    n_abstain = sum(counts.get(s, 0) for s in _ABSTAIN_STATUSES)
    n_multihop = sum(1 for r in records if bool(r.get("multihop")))

    report = {
        "log_path": str(path),
        "generated_at": utc_stamp(),
        "model_version": getattr(cfg.serving, "model_version", None),
        "api_version": getattr(cfg.serving, "api_version", None),
        "total_queries": total,
        "status_counts": counts,
        "status_rates": rates,
        "abstain_count": n_abstain,
        "abstain_rate": round(n_abstain / total, 4),
        "answered_rate": rates.get("answered", 0.0),
        "multihop_count": n_multihop,
        "multihop_rate": round(n_multihop / total, 4),
        "confidence": _summ(records, "confidence"),
        "faithfulness": _summ(records, "faithfulness"),
        "n_citations": _summ(records, "n_citations"),
        "time_range": _time_range(records),
    }

    _write_report(report)
    logger.info(
        "monitoring_report: %d queries, abstain_rate=%.3f, mean_faithfulness=%s",
        total, report["abstain_rate"], report["faithfulness"]["mean"],
    )
    return report


def drift_report(cfg: AppConfig, reference_path: str, current_path: str) -> dict:
    """Compare two query logs and flag answer-status drift via PSI.

    Builds the answer-status distribution for a trusted *reference* window and a
    *current* window, then computes the Population Stability Index between them.
    ``PSI > 0.2`` sets ``drift=True`` (the conventional "significant shift"
    threshold). Per-bucket reference/current rates are included so a reader can
    see *which* status moved.

    Either log being empty yields ``{'note': 'no logs'}`` — you cannot assess
    drift without traffic on both sides. The result is written to
    ``runs/monitoring/monitoring-<stamp>.json``.

    Parameters
    ----------
    cfg:
        Application config (echoed for provenance; not otherwise required).
    reference_path:
        Path to the baseline JSONL query log.
    current_path:
        Path to the recent JSONL query log to compare against the baseline.

    Returns
    -------
    dict
        Drift report including ``psi``, ``drift`` flag, threshold, per-bucket
        rates, and a short summary per side.
    """
    ref_path = Path(reference_path)
    cur_path = Path(current_path)
    ref_recs = _ask_records(_read_jsonl(ref_path))
    cur_recs = _ask_records(_read_jsonl(cur_path))

    if not ref_recs or not cur_recs:
        report: Dict[str, Any] = {
            "note": "no logs",
            "reference_path": str(ref_path),
            "current_path": str(cur_path),
            "reference_n": len(ref_recs),
            "current_n": len(cur_recs),
            "generated_at": utc_stamp(),
        }
        _write_report(report)
        return report

    ref_counts = _status_counts(ref_recs)
    cur_counts = _status_counts(cur_recs)
    ref_dist = _distribution(ref_counts)
    cur_dist = _distribution(cur_counts)

    psi = _psi(ref_dist, cur_dist)
    drift = psi > _PSI_DRIFT_THRESHOLD

    # Per-bucket contribution makes the headline PSI explainable: which status
    # shifted, and by how much.
    per_status = {
        s: {
            "reference_rate": round(ref_dist[i], 4),
            "current_rate": round(cur_dist[i], 4),
            "delta": round(cur_dist[i] - ref_dist[i], 4),
        }
        for i, s in enumerate(_STATUSES)
    }

    report = {
        "reference_path": str(ref_path),
        "current_path": str(cur_path),
        "generated_at": utc_stamp(),
        "reference_n": len(ref_recs),
        "current_n": len(cur_recs),
        "metric": "psi_status_distribution",
        "psi": round(psi, 6),
        "threshold": _PSI_DRIFT_THRESHOLD,
        "drift": bool(drift),
        "severity": _psi_severity(psi),
        "statuses": list(_STATUSES),
        "per_status": per_status,
        "reference": {
            "status_counts": ref_counts,
            "abstain_rate": _abstain_rate(ref_counts, len(ref_recs)),
        },
        "current": {
            "status_counts": cur_counts,
            "abstain_rate": _abstain_rate(cur_counts, len(cur_recs)),
        },
    }

    _write_report(report)
    logger.info("drift_report: psi=%.4f drift=%s (threshold=%.2f)",
                psi, drift, _PSI_DRIFT_THRESHOLD)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers shared by the two public reports
# ─────────────────────────────────────────────────────────────────────────────

def _abstain_rate(counts: Dict[str, int], total: int) -> Optional[float]:
    if total <= 0:
        return None
    return round(sum(counts.get(s, 0) for s in _ABSTAIN_STATUSES) / total, 4)


def _psi_severity(psi: float) -> str:
    """Map a PSI value to the conventional three-band severity label."""
    if psi < 0.1:
        return "none"
    if psi < _PSI_DRIFT_THRESHOLD:
        return "moderate"
    return "significant"


def _write_report(report: Dict[str, Any]) -> Optional[Path]:
    """Persist a report under ``runs/monitoring/monitoring-<stamp>.json``.

    Never raises: a monitoring read should still succeed even if the artifact
    directory is read-only.
    """
    try:
        out_dir = run_dir() / "monitoring"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"monitoring-{utc_stamp()}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report.setdefault("_artifact_path", str(out_path))
        return out_path
    except Exception as exc:  # pragma: no cover - best-effort persistence
        logger.warning("Could not write monitoring report: %s", exc)
        return None


__all__ = ["monitoring_report", "drift_report"]
