"""Rubric completeness self-check for the KBQA submission.

This module performs a *static* audit of the repository on disk and grades
each rubric requirement as ``PASS`` / ``WARN`` / ``FAIL``. It is intentionally
dependency-free (only the standard library + our own ``logging_utils``) so it
can run anywhere — including a fresh checkout with no optional deps installed —
and never crashes the autopilot.

Grading semantics
-----------------
* ``PASS`` — a required artifact is present.
* ``FAIL`` — a *required* artifact is missing.
* ``WARN`` — an *optional* (nice-to-have) artifact is missing, or a required
  one is present but suspiciously empty/degenerate.

Public API
----------
* :func:`build_checklist` — return the structured grading dict.
* :func:`write_checklist`  — run :func:`build_checklist` and persist it as JSON.

The CLI (``kbqa grade``) imports :func:`build_checklist` by name, so its
signature is load-bearing and must not change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging_utils import get_logger, utc_stamp, utc_now_iso

log = get_logger(__name__)

# Status constants (kept as plain strings so the JSON is self-describing).
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level filesystem probes — all tolerant of missing/odd paths.
# ─────────────────────────────────────────────────────────────────────────────

def _exists(root: Path, rel: str) -> bool:
    """True if ``root/rel`` exists (file *or* directory). Never raises."""
    try:
        return (root / rel).exists()
    except OSError:  # pragma: no cover - exotic path errors
        return False


def _is_dir(root: Path, rel: str) -> bool:
    try:
        return (root / rel).is_dir()
    except OSError:  # pragma: no cover
        return False


def _nonempty_file(root: Path, rel: str) -> bool:
    """True if ``root/rel`` is a file with > 0 bytes."""
    try:
        p = root / rel
        return p.is_file() and p.stat().st_size > 0
    except OSError:  # pragma: no cover
        return False


def _glob(root: Path, pattern: str) -> List[Path]:
    """Recursive glob that never raises; returns [] on any error."""
    try:
        return sorted(root.glob(pattern))
    except OSError:  # pragma: no cover
        return []


def _item(name: str, status: str, detail: str) -> Dict[str, str]:
    """Build one checklist row (and log it at an appropriate level)."""
    if status == FAIL:
        log.warning("checklist FAIL: %s — %s", name, detail)
    elif status == WARN:
        log.info("checklist WARN: %s — %s", name, detail)
    return {"name": name, "status": status, "detail": detail}


def _file_item(root: Path, name: str, rel: str, *, required: bool = True) -> Dict[str, str]:
    """Grade presence of a single file/dir.

    When missing: ``FAIL`` if required else ``WARN``.
    """
    if _exists(root, rel):
        return _item(name, PASS, f"found: {rel}")
    miss_status = FAIL if required else WARN
    qualifier = "required" if required else "optional"
    return _item(name, miss_status, f"missing {qualifier}: {rel}")


# ─────────────────────────────────────────────────────────────────────────────
# Requirement groups
# ─────────────────────────────────────────────────────────────────────────────

# Core package modules under ``src/kbqa`` that the rubric expects. Each entry is
# (human-readable name, path relative to repo root, required?).
_PACKAGE_MODULES = [
    ("Package: src/kbqa", "src/kbqa/__init__.py", True),
    ("Module: config", "src/kbqa/config.py", True),
    ("Module: cli", "src/kbqa/cli.py", True),
    ("Module: agent/rag_agent", "src/kbqa/agent/rag_agent.py", True),
    ("Module: agent/tools", "src/kbqa/agent/tools.py", True),
    ("Module: models/retriever", "src/kbqa/models/retriever.py", True),
    ("Module: models/reranker", "src/kbqa/models/reranker.py", True),
    ("Module: models/reader_extractive", "src/kbqa/models/reader_extractive.py", True),
    ("Module: index/vector_store", "src/kbqa/index/vector_store.py", True),
    ("Module: api/main", "src/kbqa/api/main.py", True),
]

# Top-level project directories.
_DIRS = [
    ("Dir: src/", "src", True),
    ("Dir: data/", "data", True),
    ("Dir: models/", "models", True),
    ("Dir: configs/", "configs", True),
    ("Dir: tests/", "tests", True),
    ("Dir: docs/", "docs", True),
    ("Dir: notebooks/", "notebooks", True),
]

# Root-level project files.
_ROOT_FILES = [
    ("File: README.md", "README.md", True),
    ("File: requirements.txt", "requirements.txt", True),
    ("File: pyproject.toml", "pyproject.toml", True),
    ("File: Dockerfile", "Dockerfile", True),
]

# Required design/spec documents (basename stem under docs/, any extension).
_DOC_STEMS = [
    "problem_definition",
    "data_description",
    "model_selection",
    "deployment",
    "agent_architecture",
    "continual_learning_monitoring",
    "privacy_robustness",
    "project_plan",
    "ethics_statement",
    "faithfulness_evaluation",
]


def _check_packages(root: Path) -> List[Dict[str, str]]:
    return [_file_item(root, name, rel, required=req) for name, rel, req in _PACKAGE_MODULES]


def _check_dirs(root: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for name, rel, req in _DIRS:
        if _is_dir(root, rel):
            items.append(_item(name, PASS, f"found dir: {rel}"))
        else:
            status = FAIL if req else WARN
            qual = "required" if req else "optional"
            items.append(_item(name, status, f"missing {qual} dir: {rel}"))
    return items


def _check_root_files(root: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for name, rel, req in _ROOT_FILES:
        if _nonempty_file(root, rel):
            items.append(_item(name, PASS, f"found: {rel}"))
        elif _exists(root, rel):
            # Present but empty — flag as WARN rather than a hard PASS.
            items.append(_item(name, WARN, f"present but empty: {rel}"))
        else:
            status = FAIL if req else WARN
            qual = "required" if req else "optional"
            items.append(_item(name, status, f"missing {qual}: {rel}"))
    return items


def _check_docs(root: Path) -> List[Dict[str, str]]:
    """Each required doc stem must exist as ``docs/<stem>.md`` (or any ext)."""
    items: List[Dict[str, str]] = []
    docs_dir = root / "docs"
    docs_present = docs_dir.is_dir()
    # Pre-index existing doc filenames (lowercased stems) for tolerant matching.
    existing_stems = set()
    if docs_present:
        for p in _glob(root, "docs/*"):
            if p.is_file():
                existing_stems.add(p.stem.lower())
    for stem in _DOC_STEMS:
        name = f"Doc: {stem}.md"
        if not docs_present:
            items.append(_item(name, FAIL, "docs/ directory absent"))
            continue
        # Accept the canonical .md or any other extension with the same stem.
        if _nonempty_file(root, f"docs/{stem}.md"):
            items.append(_item(name, PASS, f"found: docs/{stem}.md"))
        elif stem.lower() in existing_stems:
            items.append(_item(name, PASS, f"found doc with stem '{stem}' (non-.md)"))
        elif _exists(root, f"docs/{stem}.md"):
            items.append(_item(name, WARN, f"present but empty: docs/{stem}.md"))
        else:
            items.append(_item(name, FAIL, f"missing required doc: docs/{stem}.md"))
    return items


def _check_notebooks(root: Path) -> List[Dict[str, str]]:
    """At least one ``.ipynb`` somewhere under ``notebooks/``."""
    name = "Notebooks: >=1 .ipynb under notebooks/"
    if not _is_dir(root, "notebooks"):
        return [_item(name, FAIL, "notebooks/ directory absent")]
    nbs = _glob(root, "notebooks/**/*.ipynb")
    if nbs:
        sample = ", ".join(p.name for p in nbs[:3])
        more = "" if len(nbs) <= 3 else f" (+{len(nbs) - 3} more)"
        return [_item(name, PASS, f"{len(nbs)} notebook(s): {sample}{more}")]
    return [_item(name, FAIL, "no .ipynb found under notebooks/")]


def _check_tests(root: Path) -> List[Dict[str, str]]:
    """At least one ``test_*.py`` / ``*_test.py`` under ``tests/``."""
    name = "Tests: >=1 test file under tests/"
    if not _is_dir(root, "tests"):
        return [_item(name, FAIL, "tests/ directory absent")]
    tests = _glob(root, "tests/**/test_*.py") + _glob(root, "tests/**/*_test.py")
    # De-duplicate while preserving order.
    seen: set = set()
    uniq = [p for p in tests if not (p in seen or seen.add(p))]
    if uniq:
        sample = ", ".join(p.name for p in uniq[:3])
        more = "" if len(uniq) <= 3 else f" (+{len(uniq) - 3} more)"
        return [_item(name, PASS, f"{len(uniq)} test file(s): {sample}{more}")]
    # A tests/ dir with no recognisable test files is a WARN — the dir exists
    # but provides no coverage. The directory-presence check above already
    # FAILs separately if the dir is missing.
    return [_item(name, WARN, "tests/ present but no test_*.py / *_test.py found")]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_checklist(repo_root: Path) -> Dict[str, Any]:
    """Audit ``repo_root`` against the project rubric.

    Parameters
    ----------
    repo_root:
        Path to the repository root (the directory that contains ``src/``,
        ``docs/`` etc.). Accepts ``str`` or :class:`pathlib.Path`.

    Returns
    -------
    dict
        ``{'items': [{name, status, detail}, ...],
           'summary': {'PASS': int, 'WARN': int, 'FAIL': int}}``
        plus convenience metadata keys (``repo_root``, ``generated_at``,
        ``ok``). The core ``items``/``summary`` shape matches the spec exactly.
    """
    root = Path(repo_root).expanduser()
    try:
        root = root.resolve()
    except OSError:  # pragma: no cover - resolve can fail on broken paths
        pass

    items: List[Dict[str, str]] = []

    if not root.is_dir():
        # Degrade gracefully: emit a single FAIL rather than raising so the
        # autopilot can still complete and report the problem.
        log.warning("checklist: repo_root is not a directory: %s", root)
        items.append(_item("Repo root", FAIL, f"not a directory: {root}"))
    else:
        # Each group is wrapped so one unexpected error cannot sink the whole
        # audit — a failing group contributes a single FAIL item instead.
        groups = (
            ("packages", _check_packages),
            ("directories", _check_dirs),
            ("root_files", _check_root_files),
            ("docs", _check_docs),
            ("notebooks", _check_notebooks),
            ("tests", _check_tests),
        )
        for label, fn in groups:
            try:
                items.extend(fn(root))
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("checklist group '%s' failed: %s", label, exc)
                items.append(_item(f"Group: {label}", FAIL, f"check errored: {exc}"))

    summary = {
        PASS: sum(1 for it in items if it["status"] == PASS),
        WARN: sum(1 for it in items if it["status"] == WARN),
        FAIL: sum(1 for it in items if it["status"] == FAIL),
    }
    log.info(
        "checklist complete: %d PASS / %d WARN / %d FAIL",
        summary[PASS], summary[WARN], summary[FAIL],
    )
    return {
        "items": items,
        "summary": summary,
        "repo_root": str(root),
        "generated_at": utc_now_iso(),
        "ok": summary[FAIL] == 0,
    }


def write_checklist(repo_root: Path, out_path: Optional[Path] = None) -> Path:
    """Build the checklist and write it as pretty JSON.

    Parameters
    ----------
    repo_root:
        Repository root passed straight to :func:`build_checklist`.
    out_path:
        Destination JSON file. If ``None``, defaults to
        ``run_dir()/grading-<utc_stamp>/checklist.json`` per the artifact
        convention. Parent directories are created as needed.

    Returns
    -------
    pathlib.Path
        The path the JSON was written to.
    """
    report = build_checklist(repo_root)

    if out_path is None:
        # Import lazily so a misconfigured config layer cannot break the import
        # of this module (it must stay importable for the CLI in all cases).
        try:
            from ..config import run_dir
            out_path = run_dir() / f"grading-{utc_stamp()}" / "checklist.json"
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("checklist: could not resolve run_dir (%s); using CWD", exc)
            out_path = Path.cwd() / f"grading-{utc_stamp()}" / "checklist.json"

    out = Path(out_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("checklist written -> %s", out)
    except OSError as exc:  # pragma: no cover - disk/permission issues
        log.warning("checklist: failed to write %s: %s", out, exc)
    return out


__all__ = ["build_checklist", "write_checklist"]
