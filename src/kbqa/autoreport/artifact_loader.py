"""Discovery + loading of run artifacts for the auto-report generator.

The autopilot writes evaluation, benchmark, error-analysis and faithfulness
artifacts under timestamped directories::

    run_dir()/<kind>-<utc_stamp>/<name>.json

(e.g. ``runs/eval-20260626-120000/eval.json``). This module finds the *newest*
such directory for a given ``kind`` and loads its JSON payload, plus the latest
trained retriever/reader model metadata. Everything degrades gracefully: a
missing directory, an unreadable file or malformed JSON simply yields ``None``
(or ``{}`` for metadata) so report generation never crashes on partial runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import model_dir, run_dir
from ..logging_utils import get_logger
from ..models.model_registry import load_model_metadata, resolve_latest

logger = get_logger(__name__)

# Recognised artifact kinds and the directory prefix they live under.
_KINDS = ("eval", "benchmark", "error_analysis", "faithfulness")

# For each kind, the preferred JSON file name to read from inside its dir.
# We fall back to the single ``*.json`` present if the preferred name is absent.
_PREFERRED_NAME: Dict[str, str] = {
    "eval": "eval.json",
    "benchmark": "benchmark.json",
    "error_analysis": "error_analysis.json",
    "faithfulness": "faithfulness.json",
}


def _read_json(path: Path) -> Optional[dict]:
    """Read a JSON file, returning ``None`` on any failure (missing/malformed)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Could not parse artifact JSON %s: %s", path, exc)
        return None
    # Only dict payloads are meaningful for the report; coerce others to None.
    return data if isinstance(data, dict) else None


def _newest_kind_dir(kind: str) -> Optional[Path]:
    """Return the newest ``run_dir()/<kind>-*`` directory, or ``None``.

    Some kinds are written as a flat ``run_dir()/<kind>-*.json`` file instead of
    a directory; those are handled by :func:`load_latest` directly. Here we only
    consider sub-directories. Sorting is by directory name, which embeds a
    sortable ``utc_stamp`` (``YYYYMMDD-HHMMSS``).
    """
    base = run_dir()
    try:
        if not base.exists():
            return None
        candidates = [
            p for p in base.iterdir()
            if p.is_dir() and p.name.startswith(f"{kind}-")
        ]
    except OSError as exc:  # pragma: no cover - filesystem edge case
        logger.warning("Could not list run dir %s: %s", base, exc)
        return None
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def _newest_kind_file(kind: str) -> Optional[Path]:
    """Return the newest flat ``run_dir()/<kind>-*.json`` file, or ``None``."""
    base = run_dir()
    try:
        if not base.exists():
            return None
        candidates = [
            p for p in base.iterdir()
            if p.is_file() and p.name.startswith(f"{kind}-") and p.suffix == ".json"
        ]
    except OSError as exc:  # pragma: no cover - filesystem edge case
        logger.warning("Could not list run dir %s: %s", base, exc)
        return None
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_latest(kind: str) -> Optional[dict]:
    """Load the newest artifact of ``kind`` from ``run_dir()``.

    Parameters
    ----------
    kind:
        One of ``{'eval', 'benchmark', 'error_analysis', 'faithfulness'}``.

    Returns
    -------
    dict | None
        The parsed JSON payload of the most recent matching artifact, or
        ``None`` if nothing is found / readable. Both the directory layout
        (``run_dir()/<kind>-<stamp>/<name>.json``) and the flat-file layout
        (``run_dir()/<kind>-<stamp>.json``) are supported; the directory form
        wins when both newest candidates exist for the same kind.
    """
    if kind not in _KINDS:
        logger.warning("Unknown artifact kind %r (expected one of %s)", kind, _KINDS)
        return None

    newest_dir = _newest_kind_dir(kind)
    newest_file = _newest_kind_file(kind)

    # Pick whichever is newer by name; directory wins ties for parity with the
    # conventional ``<kind>-<stamp>/<name>.json`` layout.
    chosen_dir_name = newest_dir.name if newest_dir else None
    chosen_file_name = newest_file.stem if newest_file else None
    if newest_dir is not None and (
        chosen_file_name is None or chosen_dir_name >= chosen_file_name
    ):
        return _load_from_dir(kind, newest_dir)
    if newest_file is not None:
        payload = _read_json(newest_file)
        if payload is not None:
            logger.info("Loaded %s artifact from %s", kind, newest_file)
        return payload
    return None


def _load_from_dir(kind: str, directory: Path) -> Optional[dict]:
    """Load the JSON payload from a timestamped artifact directory."""
    preferred = directory / _PREFERRED_NAME.get(kind, f"{kind}.json")
    if preferred.exists():
        payload = _read_json(preferred)
        if payload is not None:
            logger.info("Loaded %s artifact from %s", kind, preferred)
            return payload

    # Fall back to the single ``*.json`` in the directory (sorted for stability).
    try:
        jsons = sorted(directory.glob("*.json"))
    except OSError as exc:  # pragma: no cover - filesystem edge case
        logger.warning("Could not glob %s: %s", directory, exc)
        return None
    for candidate in jsons:
        payload = _read_json(candidate)
        if payload is not None:
            logger.info("Loaded %s artifact from %s", kind, candidate)
            return payload
    logger.warning("No readable JSON found for %s in %s", kind, directory)
    return None


def _load_model_meta(subdir: str) -> Dict[str, Any]:
    """Best-effort load of the latest model metadata under ``model_dir()/<subdir>``."""
    try:
        base = model_dir() / subdir
        if not base.exists():
            return {}
        return load_model_metadata(resolve_latest(base)) or {}
    except Exception as exc:  # pragma: no cover - defensive: never crash report
        logger.warning("Could not load %s model metadata: %s", subdir, exc)
        return {}


def load_all_artifacts() -> Dict[str, Any]:
    """Load every artifact needed to build the auto-report.

    Returns
    -------
    dict
        A mapping with keys ``eval``, ``benchmark``, ``error_analysis`` and
        ``faithfulness`` (each ``dict | None``) plus ``retriever_meta`` and
        ``reader_meta`` (each ``dict``, possibly empty). All loading is
        best-effort: missing or malformed inputs become ``None`` / ``{}``.
    """
    artifacts: Dict[str, Any] = {kind: load_latest(kind) for kind in _KINDS}
    artifacts["retriever_meta"] = _load_model_meta("retriever")
    artifacts["reader_meta"] = _load_model_meta("reader")
    return artifacts


__all__ = ["load_latest", "load_all_artifacts"]
