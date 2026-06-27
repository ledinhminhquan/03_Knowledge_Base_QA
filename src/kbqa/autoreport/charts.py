"""Matplotlib chart generation for the KBQA auto-report / slides.

Every public function turns a slice of the merged ``artifacts`` dict (the JSON
snapshots produced by ``evaluate`` / ``benchmark`` / ``error_analysis`` /
``faithfulness_eval``) into a single PNG saved under ``out_dir`` and returns the
saved :class:`pathlib.Path` (or ``None`` when the data / matplotlib is missing).

Design rules
------------
* **Lazy, headless matplotlib.** ``matplotlib`` is imported *inside* each
  function and forced onto the non-interactive ``Agg`` backend, so importing
  this module never drags in a heavy/optional dependency and chart code never
  needs a display.
* **Degrade gracefully.** A missing optional dependency, an empty / malformed
  artifact, or a render error is logged as a warning and yields ``None`` — it
  must never crash the autopilot. ``build_all_charts`` always returns a dict and
  simply omits the charts it could not build.
* **Defensive parsing.** The analysis artifacts are read with tolerant helpers
  that accept several plausible key spellings / nesting depths, because the
  charts are decoupled from the exact producer layout.
* **No seaborn.** Pure matplotlib only.

The ``artifacts`` argument is the merged mapping the autopilot assembles, e.g.::

    {
      "eval":          {... evaluate() result ...},
      "benchmark":     {... latency benchmark ...},
      "faithfulness":  {... faithfulness eval ...},
      "error_analysis":{... error analysis ...},
    }

For robustness each helper also accepts a *flattened* artifacts dict (i.e. the
relevant sub-dict passed directly), so callers may hand either shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import artifacts_dir
from ..logging_utils import get_logger

logger = get_logger(__name__)

# A small, colour-blind-friendly palette (used cyclically by the bar charts).
_PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2", "#db2777"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _default_out_dir(out_dir: Optional[Path | str]) -> Path:
    """Resolve / create the figures output directory."""
    d = Path(out_dir) if out_dir is not None else artifacts_dir() / "submission" / "_figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _section(artifacts: Optional[Dict[str, Any]], *names: str) -> Dict[str, Any]:
    """Return the first matching sub-section of ``artifacts``.

    Accepts both the *merged* artifacts dict (``{"eval": {...}}``) and a
    *flattened* dict (the section itself). Falls back to the whole mapping so a
    flattened payload still works.
    """
    if not isinstance(artifacts, dict):
        return {}
    for name in names:
        sub = artifacts.get(name)
        if isinstance(sub, dict) and sub:
            return sub
    # Heuristic: if the top-level dict already *looks like* the section we want
    # (i.e. one of the requested keys appears nested under it), use it as-is.
    return artifacts


def _num(value: Any) -> Optional[float]:
    """Best-effort coercion to ``float``; ``None`` when not numeric."""
    if isinstance(value, bool):  # avoid treating True/False as 1/0 silently
        return None
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        # Reject NaN/inf which break matplotlib axis scaling.
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    if isinstance(value, str):
        try:
            return _num(float(value))
        except (TypeError, ValueError):
            return None
    return None


def _dig(d: Any, *keys: str) -> Any:
    """Search a (possibly nested) mapping for the first present key.

    Looks at the top level first, then one level deep into any nested dicts.
    Returns the raw value (caller coerces) or ``None``.
    """
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    for v in d.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v:
                    return v[k]
    return None


def _find_prefixed(d: Dict[str, Any], prefix: str) -> Optional[float]:
    """Find ``recall@<k>_<prefix>``-style keys without knowing ``k``.

    e.g. ``_find_prefixed(section, "hybrid")`` matches ``recall@5_hybrid``.
    """
    if not isinstance(d, dict):
        return None
    suffix = "_" + prefix
    for key, val in d.items():
        if isinstance(key, str) and key.startswith("recall@") and key.endswith(suffix):
            n = _num(val)
            if n is not None:
                return n
    return None


def _percentish(value: Optional[float]) -> Optional[float]:
    """Normalise a rate/percentage onto a 0–100 scale.

    Values in ``[0, 1]`` are treated as fractions and scaled ×100; values
    already > 1 (e.g. an F1 of 73.4) are assumed to be percentages already.
    """
    if value is None:
        return None
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def _save(fig, out_dir: Path, name: str) -> Optional[Path]:
    """Persist ``fig`` to ``out_dir/name`` and close it; return the path."""
    try:
        path = out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        return path
    except Exception as exc:  # pragma: no cover - disk / backend failure
        logger.warning("Failed to save chart %s: %s", name, exc)
        return None
    finally:
        try:
            import matplotlib.pyplot as plt  # noqa: WPS433 (local import by design)

            plt.close(fig)
        except Exception:
            pass


def _import_mpl():
    """Lazily import matplotlib on the headless ``Agg`` backend.

    Returns the ``pyplot`` module, or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no DISPLAY / Tk required
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # ImportError or backend failure
        logger.warning("matplotlib unavailable (%s); skipping chart.", exc)
        return None


def _bar_chart(
    out_dir: Path,
    filename: str,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    ylabel: str = "",
    ylim: Optional[Tuple[float, float]] = None,
    value_fmt: str = "{:.1f}",
    colors: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """Render a labelled vertical bar chart and save it.

    Shared by all four public chart functions to keep styling consistent.
    """
    plt = _import_mpl()
    if plt is None:
        return None
    if not labels or not values:
        logger.warning("No data for chart %r; skipping.", title)
        return None
    try:
        colors = list(colors or _PALETTE)
        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]

        fig, ax = plt.subplots(figsize=(max(5.0, 1.25 * len(labels) + 2.0), 4.2))
        x = range(len(labels))
        bars = ax.bar(list(x), list(values), color=bar_colors, width=0.62, zorder=3)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9, rotation=0 if len(labels) <= 5 else 20,
                           ha="center" if len(labels) <= 5 else "right")
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            top = max(values) if values else 1.0
            ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)

        # Numeric labels above each bar.
        for rect, val in zip(bars, values):
            ax.annotate(
                value_fmt.format(val),
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, fontweight="medium",
            )
        fig.tight_layout()
        return _save(fig, out_dir, filename)
    except Exception as exc:  # pragma: no cover - render-time failure
        logger.warning("Failed to render chart %r: %s", title, exc)
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def _grouped_bar_chart(
    out_dir: Path,
    filename: str,
    title: str,
    groups: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float]]],
    *,
    ylabel: str = "",
    ylim: Optional[Tuple[float, float]] = None,
    value_fmt: str = "{:.1f}",
) -> Optional[Path]:
    """Render a grouped (multi-series) vertical bar chart and save it."""
    plt = _import_mpl()
    if plt is None:
        return None
    if not groups or not series:
        logger.warning("No data for grouped chart %r; skipping.", title)
        return None
    try:
        n_series = len(series)
        total_width = 0.8
        bar_w = total_width / max(1, n_series)
        x = list(range(len(groups)))

        fig, ax = plt.subplots(figsize=(max(5.5, 1.6 * len(groups) + 2.0), 4.4))
        for si, (label, vals) in enumerate(series):
            offsets = [xi - total_width / 2 + bar_w * (si + 0.5) for xi in x]
            bars = ax.bar(
                offsets, list(vals), width=bar_w * 0.95,
                color=_PALETTE[si % len(_PALETTE)], label=label, zorder=3,
            )
            for rect, val in zip(bars, vals):
                if val is None:
                    continue
                ax.annotate(
                    value_fmt.format(val),
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5,
                )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(fontsize=8.5, frameon=False, ncol=min(n_series, 3))

        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            flat = [v for _, vals in series for v in vals if v is not None]
            top = max(flat) if flat else 1.0
            ax.set_ylim(0, top * 1.2 if top > 0 else 1.0)
        fig.tight_layout()
        return _save(fig, out_dir, filename)
    except Exception as exc:  # pragma: no cover - render-time failure
        logger.warning("Failed to render grouped chart %r: %s", title, exc)
        try:
            plt.close("all")
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public chart functions
# ─────────────────────────────────────────────────────────────────────────────

def metrics_chart(artifacts: Dict[str, Any], out_dir: Optional[Path | str] = None) -> Optional[Path]:
    """Headline eval metrics as a bar chart (0–100 scale).

    Plots, when present:
      * ``recall@5_hybrid`` (×100),
      * ``answer_f1`` (already 0–100, or the raw F1 ×100),
      * ``abstain_recall`` (×100).

    If both hybrid and BM25 retrieval recall are available, a second grouped
    "Retrieval recall" chart (hybrid vs BM25) is also produced — but this
    function returns the **primary** metrics PNG path.
    """
    out = _default_out_dir(out_dir)
    ev = _section(artifacts, "eval", "evaluate", "evaluation")
    summary = ev.get("summary") if isinstance(ev.get("summary"), dict) else {}

    # Hybrid recall (prefer summary, then retrieval block, k-agnostic match).
    recall_hybrid = _percentish(_num(
        _dig(summary, "recall@5_hybrid")
        or _find_prefixed(summary, "hybrid")
        or _find_prefixed(ev.get("retrieval", {}), "hybrid")
        or _find_prefixed(ev, "hybrid")
    ))
    # Answer F1 (summary 'answer_f1' or answer.f1).
    answer_f1 = _num(_dig(summary, "answer_f1") or _dig(ev.get("answer", {}), "f1", "answer_f1") or _dig(ev, "answer_f1", "f1"))
    answer_f1 = _percentish(answer_f1)
    # Abstain recall.
    abstain = _percentish(_num(
        _dig(summary, "abstain_recall")
        or _dig(ev.get("abstention", {}), "abstain_recall")
        or _dig(ev, "abstain_recall")
    ))

    labels: List[str] = []
    values: List[float] = []
    for label, val in (("Recall@5 (hybrid)", recall_hybrid),
                       ("Answer F1", answer_f1),
                       ("Abstain recall", abstain)):
        if val is not None:
            labels.append(label)
            values.append(round(val, 2))

    if not values:
        logger.warning("metrics_chart: no eval metrics found; skipping.")
        return None

    primary = _bar_chart(
        out, "metrics.png", "End-to-end QA Metrics",
        labels, values, ylabel="score (%)", ylim=(0, 100), value_fmt="{:.1f}",
    )

    # Optional companion: retrieval recall hybrid vs bm25.
    retr = ev.get("retrieval", {}) if isinstance(ev.get("retrieval"), dict) else ev
    rh = _percentish(_num(_find_prefixed(retr, "hybrid") or _dig(summary, "recall@5_hybrid")))
    rb = _percentish(_num(_find_prefixed(retr, "bm25")))
    if rh is not None and rb is not None:
        _grouped_bar_chart(
            out, "metrics_retrieval.png", "Retrieval Recall@5: Hybrid vs BM25",
            ["Recall@5"],
            [("Hybrid", [round(rh, 2)]), ("BM25", [round(rb, 2)])],
            ylabel="recall (%)", ylim=(0, 100),
        )

    return primary


def latency_chart(artifacts: Dict[str, Any], out_dir: Optional[Path | str] = None) -> Optional[Path]:
    """Latency percentiles (p50/p95/p99) for the ``ask`` and ``search`` paths.

    Reads the benchmark artifact, tolerating a few layouts::

        {"ask": {"p50":..,"p95":..,"p99":..}, "search": {...}}
        {"latency": {"ask": {...}, "search": {...}}}
        {"ask": {"p50_ms":..}, ...}   # *_ms suffix accepted too

    Produces a grouped bar chart (groups = percentiles, series = ask/search).
    """
    out = _default_out_dir(out_dir)
    bench = _section(artifacts, "benchmark", "latency", "benchmarks")
    # Allow a 'latency' wrapper inside the benchmark artifact.
    if isinstance(bench.get("latency"), dict):
        inner = bench["latency"]
        if isinstance(inner.get("ask"), dict) or isinstance(inner.get("search"), dict):
            bench = inner

    percentiles = ["p50", "p95", "p99"]

    def _series_for(op: str) -> Optional[List[Optional[float]]]:
        block = bench.get(op)
        if not isinstance(block, dict):
            return None
        out_vals: List[Optional[float]] = []
        any_val = False
        for p in percentiles:
            v = _num(_dig(block, p, f"{p}_ms", p.upper(), f"{p}_latency_ms"))
            out_vals.append(v)
            any_val = any_val or v is not None
        return out_vals if any_val else None

    series: List[Tuple[str, Sequence[float]]] = []
    for op, nice in (("ask", "ask"), ("search", "search")):
        vals = _series_for(op)
        if vals is not None:
            # Replace missing percentiles with 0.0 so the group still renders.
            series.append((nice, [v if v is not None else 0.0 for v in vals]))

    if not series:
        logger.warning("latency_chart: no benchmark latency data found; skipping.")
        return None

    return _grouped_bar_chart(
        out, "latency.png", "Latency Percentiles (ms)",
        [p.upper() for p in percentiles], series,
        ylabel="latency (ms)", value_fmt="{:.0f}",
    )


def faithfulness_chart(artifacts: Dict[str, Any], out_dir: Optional[Path | str] = None) -> Optional[Path]:
    """Groundedness summary: grounded% / hallucination rate / abstain rate.

    Reads the faithfulness artifact, tolerating fraction-or-percentage values
    and several key spellings (``grounded``/``grounded_rate``/``grounded_pct``,
    ``hallucination_rate``/``hallucinations``, ``abstain_rate``/``abstention_rate``).
    """
    out = _default_out_dir(out_dir)
    fa = _section(artifacts, "faithfulness", "groundedness", "faithfulness_eval")

    grounded = _percentish(_num(_dig(
        fa, "grounded", "grounded_rate", "grounded_pct", "grounded_percent",
        "grounded_ratio", "support_rate", "supported_rate",
    )))
    halluc = _percentish(_num(_dig(
        fa, "hallucination_rate", "hallucination", "hallucinations",
        "hallucination_pct", "ungrounded_rate", "unsupported_rate",
    )))
    abstain = _percentish(_num(_dig(
        fa, "abstain_rate", "abstention_rate", "abstain", "abstained_rate",
        "abstain_pct", "abstention",
    )))

    labels: List[str] = []
    values: List[float] = []
    colors: List[str] = []
    for label, val, col in (("Grounded", grounded, "#16a34a"),
                            ("Hallucination", halluc, "#dc2626"),
                            ("Abstain", abstain, "#2563eb")):
        if val is not None:
            labels.append(label)
            values.append(round(val, 2))
            colors.append(col)

    if not values:
        logger.warning("faithfulness_chart: no faithfulness data found; skipping.")
        return None

    return _bar_chart(
        out, "faithfulness.png", "Answer Faithfulness",
        labels, values, ylabel="rate (%)", ylim=(0, 100),
        value_fmt="{:.1f}", colors=colors,
    )


def error_breakdown_chart(artifacts: Dict[str, Any], out_dir: Optional[Path | str] = None) -> Optional[Path]:
    """Bar chart of error-category counts from the error-analysis artifact.

    Accepts the counts under any of ``categories`` / ``error_categories`` /
    ``breakdown`` / ``counts`` / ``errors`` (a mapping ``label -> count``), or a
    flat mapping of label→int at the top level as a last resort.
    """
    out = _default_out_dir(out_dir)
    ea = _section(artifacts, "error_analysis", "errors", "error_breakdown")

    counts: Dict[str, Any] = {}
    for key in ("categories", "error_categories", "breakdown", "counts",
                "by_category", "error_types", "errors", "category_counts"):
        candidate = ea.get(key)
        if isinstance(candidate, dict) and candidate:
            counts = candidate
            break

    # Last resort: treat numeric top-level entries as the counts themselves.
    if not counts:
        numeric = {k: _num(v) for k, v in ea.items() if isinstance(k, str)}
        counts = {k: v for k, v in numeric.items() if v is not None}

    # Coerce + drop non-numeric, sort descending by count for readability.
    pairs: List[Tuple[str, float]] = []
    for label, val in counts.items():
        n = _num(val)
        if n is not None:
            pairs.append((str(label).replace("_", " "), n))
    pairs.sort(key=lambda kv: kv[1], reverse=True)

    if not pairs:
        logger.warning("error_breakdown_chart: no error categories found; skipping.")
        return None

    labels = [p[0] for p in pairs]
    values = [round(p[1], 2) for p in pairs]
    # Counts are integers; show them without decimals.
    integral = all(abs(v - round(v)) < 1e-9 for v in values)
    fmt = "{:.0f}" if integral else "{:.1f}"

    return _bar_chart(
        out, "error_breakdown.png", "Error Analysis by Category",
        labels, values, ylabel="count", value_fmt=fmt,
    )


def build_all_charts(artifacts: Dict[str, Any], out_dir: Optional[Path | str] = None) -> Dict[str, str]:
    """Build every chart that has data; return ``{name: png_path}``.

    Never raises: each chart is attempted independently and any failure /
    missing-data case is simply omitted from the returned mapping. Paths are
    stringified absolute paths for easy embedding in the PDF / PPTX builders.
    """
    out = _default_out_dir(out_dir)
    builders = (
        ("metrics", metrics_chart),
        ("latency", latency_chart),
        ("faithfulness", faithfulness_chart),
        ("error_breakdown", error_breakdown_chart),
    )
    result: Dict[str, str] = {}
    for name, fn in builders:
        try:
            path = fn(artifacts, out_dir=out)
        except Exception as exc:  # pragma: no cover - safety net per chart
            logger.warning("build_all_charts: %s chart failed: %s", name, exc)
            path = None
        if path is not None:
            result[name] = str(Path(path).resolve())

    # Surface the optional retrieval companion chart if metrics produced it.
    companion = out / "metrics_retrieval.png"
    if companion.exists():
        result.setdefault("metrics_retrieval", str(companion.resolve()))

    logger.info("build_all_charts: produced %d chart(s) -> %s", len(result), out)
    return result


__all__ = [
    "metrics_chart",
    "latency_chart",
    "faithfulness_chart",
    "error_breakdown_chart",
    "build_all_charts",
]
