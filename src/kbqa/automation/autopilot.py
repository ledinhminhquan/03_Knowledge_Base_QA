"""One-button autopilot for the KBQA system.

Runs the full project pipeline end-to-end and assembles a graded submission
bundle, all from a single call:

    build demo KB → (optionally) train retriever + reader → evaluate →
    latency benchmark → error analysis → faithfulness eval →
    PDF report → PPTX slides → rubric self-check →
    submission manifest + zipped bundle

Design constraints
------------------
* **Never crash.** Every step runs inside its own ``try/except`` and records a
  per-step status; a failure (missing optional dependency, untrained model,
  unavailable dataset, or a sibling module that is not yet present) logs a
  warning and the pipeline continues with the next step.
* **Lazy imports.** Heavy modules (training, analysis, autoreport) and their
  transitive deps (``torch``, ``matplotlib``, ``reportlab``, ``python-pptx``,
  ``datasets``, ``sentence_transformers``) are imported *inside* the relevant
  step, so an environment that lacks one of them still runs the rest.
* **Best-effort artifacts.** The report/slides are copied into the submission
  directory only if they were produced; the zip is built from whatever exists.

The single public entrypoint is :func:`run_autopilot`.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..config import AppConfig, artifacts_dir, ensure_dirs, run_dir
from ..logging_utils import get_logger, utc_now_iso, utc_stamp

logger = get_logger(__name__)

__all__ = ["run_autopilot"]


# ─────────────────────────────────────────────────────────────────────────────
# Step runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_step(report: Dict[str, Any], name: str, fn: Callable[[], Any]) -> Any:
    """Execute one pipeline step, capturing status/timing/errors.

    Records ``report[name] = {status, seconds, ...}`` and returns the step's
    result (or ``None`` on failure). Any exception is swallowed so the autopilot
    keeps going — this is the whole point of the one-button pipeline.
    """
    started = time.perf_counter()
    logger.info("autopilot ▶ %s", name)
    try:
        result = fn()
        elapsed = round(time.perf_counter() - started, 3)
        entry: Dict[str, Any] = {"status": "ok", "seconds": elapsed}
        if isinstance(result, dict):
            # Keep the manifest small: store a compact, JSON-safe summary only.
            entry["summary"] = _summarize(result)
        elif result is not None:
            entry["summary"] = _jsonable(result)
        report[name] = entry
        logger.info("autopilot ✓ %s (%.2fs)", name, elapsed)
        return result
    except Exception as exc:  # noqa: BLE001 — deliberately broad: never crash
        elapsed = round(time.perf_counter() - started, 3)
        report[name] = {
            "status": "error",
            "seconds": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
        }
        logger.warning("autopilot ✗ %s failed: %s", name, exc)
        logger.debug("%s", traceback.format_exc())
        return None


def _jsonable(value: Any) -> Any:
    """Best-effort coercion of a value into something ``json.dumps`` accepts."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _summarize(result: Dict[str, Any], max_keys: int = 24) -> Dict[str, Any]:
    """Trim a step result down to a compact, JSON-safe dict for the manifest."""
    out: Dict[str, Any] = {}
    for i, (k, v) in enumerate(result.items()):
        if i >= max_keys:
            out["_truncated"] = True
            break
        sv = _jsonable(v)
        # Avoid embedding large nested blobs in the manifest.
        if isinstance(sv, list) and len(sv) > 12:
            sv = sv[:12] + ["…(+%d)" % (len(sv) - 12)]
        out[str(k)] = sv
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run_autopilot(cfg: AppConfig, title: str = None, author: str = None,
                  train: bool = True, limit: int = None) -> Dict:
    """Run the full KBQA pipeline and assemble a submission bundle.

    Parameters
    ----------
    cfg
        The loaded :class:`AppConfig`.
    title, author
        Overrides for the report/slides cover page. Default to the values on
        ``cfg`` (``project_title`` / ``author``).
    train
        When ``True`` (default) fine-tune the retriever and reader before
        evaluation. Set ``False`` to evaluate the base models only — much faster
        and dependency-light.
    limit
        Optional cap on corpus/dataset size, threaded through to KB building,
        training and evaluation to keep smoke runs fast.

    Returns
    -------
    dict
        ``{"steps": {<name>: {status, seconds, ...}}, "submission_dir": str|None,
        "submission_zip": str|None, "manifest": str|None, "ok": int, "failed":
        int, "title": ..., "author": ..., "trained": bool}``.
    """
    ensure_dirs()
    title = title or cfg.project_title
    author = author or cfg.author
    stamp = utc_stamp()
    started_at = utc_now_iso()

    report: Dict[str, Any] = {}

    # ── 1. Build the demo knowledge base ─────────────────────────────────────
    def _step_kb():
        from ..data.corpus import build_demo_kb
        retriever = build_demo_kb(cfg, limit_corpus=limit)
        n = len(retriever.store) if getattr(retriever, "store", None) is not None else 0
        return {"n_passages": int(n)}

    _run_step(report, "build_kb", _step_kb)

    # ── 2. Train retriever + reader (optional) ───────────────────────────────
    if train:
        def _step_train_retriever():
            from ..training.train_retriever import train_retriever
            return train_retriever(cfg, limit=limit)

        def _step_train_reader():
            from ..training.train_reader import train_reader
            return train_reader(cfg, limit=limit)

        _run_step(report, "train_retriever", _step_train_retriever)
        _run_step(report, "train_reader", _step_train_reader)
    else:
        report["train_retriever"] = {"status": "skipped", "seconds": 0.0}
        report["train_reader"] = {"status": "skipped", "seconds": 0.0}

    # ── 3. End-to-end evaluation ─────────────────────────────────────────────
    def _step_evaluate():
        from ..training.evaluate import evaluate
        return evaluate(cfg, limit=(limit or 100))

    _run_step(report, "evaluate", _step_evaluate)

    # ── 4. Latency benchmark ─────────────────────────────────────────────────
    def _step_benchmark():
        from ..analysis.latency import benchmark
        return benchmark(cfg, n=40, warmup=4)

    _run_step(report, "benchmark", _step_benchmark)

    # ── 5. Error analysis ────────────────────────────────────────────────────
    def _step_error_analysis():
        from ..analysis.error_analysis import error_analysis
        return error_analysis(cfg, limit=(limit or 100))

    _run_step(report, "error_analysis", _step_error_analysis)

    # ── 6. Faithfulness / groundedness evaluation ────────────────────────────
    def _step_faithfulness():
        from ..analysis.faithfulness import faithfulness_eval
        return faithfulness_eval(cfg, limit=(limit or 100))

    _run_step(report, "faithfulness", _step_faithfulness)

    # ── 7. PDF report ────────────────────────────────────────────────────────
    report_path_holder: Dict[str, Optional[Path]] = {"report": None, "slides": None}

    def _step_report():
        from ..autoreport.report_pdf import generate_report
        p = generate_report(cfg, title=title, author=author)
        if p:
            report_path_holder["report"] = Path(p)
        return {"path": str(p) if p else None}

    _run_step(report, "generate_report", _step_report)

    # ── 8. PPTX slides ───────────────────────────────────────────────────────
    def _step_slides():
        from ..autoreport.slides_pptx import generate_slides
        p = generate_slides(cfg, title=title, author=author)
        if p:
            report_path_holder["slides"] = Path(p)
        return {"path": str(p) if p else None}

    _run_step(report, "generate_slides", _step_slides)

    # ── 9. Rubric self-check (grading checklist) ─────────────────────────────
    checklist_holder: Dict[str, Any] = {"checklist": None}

    def _step_checklist():
        from ..grading.checklist import build_checklist
        # Repo root is four levels up: .../src/kbqa/automation/autopilot.py
        repo_root = Path(__file__).resolve().parents[3]
        result = build_checklist(repo_root)
        checklist_holder["checklist"] = result
        return result

    _run_step(report, "grading_checklist", _step_checklist)

    # ── Assemble the submission bundle (manifest + copies + zip) ─────────────
    submission_dir: Optional[Path] = None
    submission_zip: Optional[Path] = None
    manifest_path: Optional[Path] = None

    def _step_bundle():
        nonlocal submission_dir, submission_zip, manifest_path

        sub_root = artifacts_dir() / "submission"
        submission_dir = sub_root / f"submission-{stamp}"
        submission_dir.mkdir(parents=True, exist_ok=True)

        copied: Dict[str, Optional[str]] = {"report.pdf": None, "slides.pptx": None}

        # Copy the report + slides into the bundle if they were produced.
        for src_key, dst_name in (("report", "report.pdf"), ("slides", "slides.pptx")):
            src = report_path_holder.get(src_key)
            try:
                if src and Path(src).exists():
                    dst = submission_dir / dst_name
                    shutil.copy2(str(src), str(dst))
                    copied[dst_name] = str(dst)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not copy %s into bundle: %s", dst_name, exc)

        # Build the submission manifest (embeds the grading checklist).
        manifest = {
            "project_title": title,
            "author": author,
            "model_version": cfg.serving.model_version,
            "api_version": cfg.serving.api_version,
            "created_at": started_at,
            "finished_at": utc_now_iso(),
            "stamp": stamp,
            "trained": bool(train),
            "limit": limit,
            "steps": report,
            "artifacts": {
                "report_pdf": copied["report.pdf"],
                "slides_pptx": copied["slides.pptx"],
            },
            "grading_checklist": _jsonable(checklist_holder.get("checklist")),
            "environment": _safe_environment(),
        }
        manifest_path = submission_dir / "submission_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Zip the entire bundle directory.
        submission_zip = sub_root / f"submission-{stamp}.zip"
        with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(submission_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(submission_dir))

        return {
            "submission_dir": str(submission_dir),
            "submission_zip": str(submission_zip),
            "manifest": str(manifest_path),
            "files": [p.name for p in sorted(submission_dir.iterdir())],
        }

    _run_step(report, "bundle", _step_bundle)

    # ── Mirror the manifest into runs/ for discoverability ───────────────────
    def _step_run_snapshot():
        out = run_dir() / f"autopilot-{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "title": title, "author": author, "trained": bool(train), "limit": limit,
            "started_at": started_at, "finished_at": utc_now_iso(),
            "submission_dir": str(submission_dir) if submission_dir else None,
            "submission_zip": str(submission_zip) if submission_zip else None,
            "steps": report,
        }
        (out / "autopilot.json").write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {"path": str(out / "autopilot.json")}

    _run_step(report, "run_snapshot", _step_run_snapshot)

    # ── Final tally ──────────────────────────────────────────────────────────
    n_ok = sum(1 for v in report.values() if isinstance(v, dict) and v.get("status") == "ok")
    n_failed = sum(1 for v in report.values() if isinstance(v, dict) and v.get("status") == "error")
    logger.info("autopilot done: %d ok, %d failed -> %s",
                n_ok, n_failed, submission_dir or "(no bundle)")

    return {
        "title": title,
        "author": author,
        "trained": bool(train),
        "limit": limit,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "steps": report,
        "ok": n_ok,
        "failed": n_failed,
        "submission_dir": str(submission_dir) if submission_dir else None,
        "submission_zip": str(submission_zip) if submission_zip else None,
        "manifest": str(manifest_path) if manifest_path else None,
    }


def _safe_environment() -> Dict[str, Any]:
    """Best-effort environment snapshot; never raises."""
    try:
        from ..models.model_registry import environment_snapshot
        return environment_snapshot()
    except Exception:  # noqa: BLE001
        import platform
        return {"python": platform.python_version(), "platform": platform.platform()}
