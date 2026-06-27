"""Automatic PDF report generator for the KBQA system.

Assembles a 10–15 page submission-quality PDF from three sources:

1. **Narrative** — the project's design documents under ``docs/`` (problem
   definition, data, model selection, agent architecture, deployment, continual
   learning, privacy/robustness, ethics, project plan, faithfulness eval). Each
   markdown file is lightly *stripped* into ReportLab flowables (headings,
   paragraphs, bullet/number lists, GFM tables, code blocks, blockquotes).
2. **Charts** — PNG figures produced by :mod:`kbqa.autoreport.charts`
   (``build_all_charts(load_all_artifacts())``). Imported defensively: if the
   sibling module or its heavy deps (matplotlib) are missing, the report is
   still produced without the figures.
3. **Metrics** — the most recent ``runs/eval-*/eval.json`` snapshot, rendered
   as a compact results table on the title/overview pages.

Design rules honoured here:
    * Heavy/optional dependencies (``reportlab``, ``matplotlib`` via charts) are
      imported **lazily inside functions** and every failure degrades
      gracefully — a missing optional dep must never crash the autopilot.
    * If ``reportlab`` itself is unavailable we fall back to writing a Markdown
      digest (``report.md``) so the pipeline still yields a deliverable.
    * Output lands under ``artifacts_dir()/submission/submission-<stamp>/`` and
      the resulting :class:`pathlib.Path` is returned.

Public API:
    generate_report(cfg, title=None, author=None, out_path=None) -> pathlib.Path
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import AppConfig, artifacts_dir, run_dir
from ..logging_utils import get_logger, utc_now_iso, utc_stamp

logger = get_logger(__name__)

__all__ = ["generate_report"]


# ─────────────────────────────────────────────────────────────────────────────
# Document set & ordering
# ─────────────────────────────────────────────────────────────────────────────

# (doc stem, human-friendly section title). Order defines the report flow.
_DOC_SECTIONS: List[Tuple[str, str]] = [
    ("problem_definition", "Problem Definition"),
    ("data_description", "Data Description"),
    ("model_selection", "Model Selection"),
    ("agent_architecture", "Agent Architecture"),
    ("faithfulness_evaluation", "Faithfulness & Evaluation"),
    ("deployment", "Deployment"),
    ("continual_learning_monitoring", "Continual Learning & Monitoring"),
    ("privacy_robustness", "Privacy & Robustness"),
    ("ethics_statement", "Ethics Statement"),
    ("project_plan", "Project Plan"),
]


def _docs_dir() -> Path:
    """Locate the repository ``docs/`` directory.

    The package lives at ``<repo>/src/kbqa``; docs live at ``<repo>/docs``.
    Fall back to a relative ``docs`` if the layout differs.
    """
    here = Path(__file__).resolve()
    # .../src/kbqa/autoreport/report_pdf.py -> parents[3] == <repo>
    for up in (3, 2, 4):
        try:
            candidate = here.parents[up] / "docs"
        except IndexError:
            continue
        if candidate.is_dir():
            return candidate
    return Path("docs")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown → lightweight block model
# ─────────────────────────────────────────────────────────────────────────────
#
# We do NOT need a full markdown engine. We parse line-by-line into a small list
# of typed blocks that the PDF builder turns into ReportLab flowables. Keeping
# the intermediate representation explicit also makes the .md fallback trivial.

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_ULIST_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_OLIST_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_HR_RE = re.compile(r"^\s*([-*_])\s*(?:\1\s*){2,}$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


def _strip_inline(text: str) -> str:
    """Reduce inline markdown to plain text suitable for ReportLab paragraphs.

    Removes emphasis/code markers and link syntax, and HTML-escapes the result
    so stray ``<`` / ``&`` cannot break ReportLab's mini-HTML paragraph parser.
    Bold/italic are then re-applied as ``<b>``/``<i>`` tags on a *clean* string.
    """
    s = text
    # Images ![alt](url) -> alt ; links [txt](url) -> txt
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    # Capture emphasis spans before escaping, using placeholders.
    # Bold (**x** or __x__)
    s = re.sub(r"(\*\*|__)(.+?)\1", lambda m: "\x00B\x00" + m.group(2) + "\x00/B\x00", s)
    # Italic (*x* or _x_) — avoid matching list bullets already consumed.
    s = re.sub(r"(?<!\w)([*_])(?!\s)(.+?)(?<!\s)\1(?!\w)",
               lambda m: "\x00I\x00" + m.group(2) + "\x00/I\x00", s)
    # Inline code `x`
    s = re.sub(r"`([^`]+)`", lambda m: "\x00C\x00" + m.group(1) + "\x00/C\x00", s)
    # Now escape HTML-special chars on the remaining literal text.
    s = html.escape(s, quote=False)
    # Restore emphasis placeholders as ReportLab tags.
    s = (s.replace("\x00B\x00", "<b>").replace("\x00/B\x00", "</b>")
          .replace("\x00I\x00", "<i>").replace("\x00/I\x00", "</i>")
          .replace("\x00C\x00", '<font face="Courier">').replace("\x00/C\x00", "</font>"))
    return s.strip()


def _parse_markdown(text: str) -> List[Dict[str, Any]]:
    """Parse markdown into an ordered list of typed blocks.

    Block kinds: ``heading`` (level,text), ``para`` (text), ``ulist``/``olist``
    (items), ``table`` (rows), ``code`` (text), ``quote`` (text), ``hr``.
    Best-effort and defensive: any unparseable construct falls back to a
    paragraph so no content is lost.
    """
    blocks: List[Dict[str, Any]] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i, n = 0, len(lines)

    para_buf: List[str] = []
    list_buf: List[str] = []
    list_kind: Optional[str] = None

    def flush_para() -> None:
        if para_buf:
            joined = " ".join(s.strip() for s in para_buf).strip()
            if joined:
                blocks.append({"kind": "para", "text": joined})
            para_buf.clear()

    def flush_list() -> None:
        nonlocal list_kind
        if list_buf:
            blocks.append({"kind": list_kind or "ulist", "items": list(list_buf)})
            list_buf.clear()
        list_kind = None

    while i < n:
        raw = lines[i]
        line = raw.rstrip()

        # Fenced code block ``` ... ```
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            flush_para()
            flush_list()
            fence = line.lstrip()[:3]
            body: List[str] = []
            i += 1
            while i < n and not lines[i].lstrip().startswith(fence):
                body.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code = "\n".join(body).rstrip("\n")
            if code.strip():
                blocks.append({"kind": "code", "text": code})
            continue

        # Blank line — paragraph/list separator.
        if not line.strip():
            flush_para()
            flush_list()
            i += 1
            continue

        # Horizontal rule.
        if _HR_RE.match(line):
            flush_para()
            flush_list()
            blocks.append({"kind": "hr"})
            i += 1
            continue

        # Heading.
        m = _HEADING_RE.match(line)
        if m:
            flush_para()
            flush_list()
            level = len(m.group(1))
            blocks.append({"kind": "heading", "level": level, "text": m.group(2).strip()})
            i += 1
            continue

        # GFM table: a header row followed by a separator row of dashes.
        if "|" in line and (i + 1) < n and _TABLE_SEP_RE.match(lines[i + 1]):
            flush_para()
            flush_list()
            rows: List[List[str]] = [_split_table_row(line)]
            i += 2  # skip header + separator
            while i < n and "|" in lines[i] and lines[i].strip():
                if _TABLE_SEP_RE.match(lines[i]):
                    i += 1
                    continue
                rows.append(_split_table_row(lines[i]))
                i += 1
            blocks.append({"kind": "table", "rows": rows})
            continue

        # Blockquote.
        if line.lstrip().startswith(">"):
            flush_para()
            flush_list()
            quote: List[str] = []
            while i < n and lines[i].lstrip().startswith(">"):
                quote.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            blocks.append({"kind": "quote", "text": " ".join(q.strip() for q in quote).strip()})
            continue

        # Unordered list item.
        mu = _ULIST_RE.match(line)
        if mu:
            flush_para()
            if list_kind not in (None, "ulist"):
                flush_list()
            list_kind = "ulist"
            list_buf.append(mu.group(1).strip())
            i += 1
            continue

        # Ordered list item.
        mo = _OLIST_RE.match(line)
        if mo:
            flush_para()
            if list_kind not in (None, "olist"):
                flush_list()
            list_kind = "olist"
            list_buf.append(mo.group(1).strip())
            i += 1
            continue

        # Continuation of a list item (indented wrap) — append to last item.
        if list_buf and (raw.startswith("  ") or raw.startswith("\t")):
            list_buf[-1] = (list_buf[-1] + " " + line.strip()).strip()
            i += 1
            continue

        # Default: paragraph text.
        flush_list()
        para_buf.append(line)
        i += 1

    flush_para()
    flush_list()
    return blocks


def _split_table_row(line: str) -> List[str]:
    """Split a markdown table row into trimmed cells."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    # Split on unescaped pipes.
    cells = re.split(r"(?<!\\)\|", s)
    return [c.replace("\\|", "|").strip() for c in cells]


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading (charts + eval metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _latest_eval() -> Optional[Dict[str, Any]]:
    """Return the most recent ``runs/eval-*/eval.json`` payload, or None."""
    try:
        rd = run_dir()
        if not rd.exists():
            return None
        evals = sorted(rd.glob("eval-*/eval.json"), key=lambda p: p.stat().st_mtime)
        if not evals:
            # Some pipelines nest eval under other run dirs — search broadly.
            evals = sorted(rd.glob("**/eval.json"), key=lambda p: p.stat().st_mtime)
        if not evals:
            return None
        return json.loads(evals[-1].read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — never crash report generation
        logger.warning("Could not load eval artifact: %s", exc)
        return None


def _collect_charts() -> List[Tuple[str, Path]]:
    """Build/collect chart PNGs via the sibling ``charts`` module, defensively.

    Returns a list of ``(label, path)`` for existing PNG files. Any failure
    (missing module, missing matplotlib, bad return type) yields an empty list.
    """
    try:
        from . import charts as _charts  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.warning("charts module unavailable (%s); report will omit figures.", exc)
        return []

    builder = getattr(_charts, "build_all_charts", None)
    if not callable(builder):
        logger.warning("charts.build_all_charts missing; report will omit figures.")
        return []

    # ``load_all_artifacts`` may live in ``charts`` or in ``artifact_loader``.
    loader = getattr(_charts, "load_all_artifacts", None)
    if not callable(loader):
        try:
            from .artifact_loader import load_all_artifacts as loader  # type: ignore
        except Exception:  # noqa: BLE001
            loader = None

    # Always pass an artifacts dict (it is a required positional arg). An empty
    # dict is fine — the builder degrades gracefully on missing artifacts.
    artifacts: Any = {}
    if callable(loader):
        try:
            loaded = loader()
            if loaded is not None:
                artifacts = loaded
        except Exception as exc:  # noqa: BLE001
            logger.warning("load_all_artifacts failed (%s); using empty artifacts.", exc)

    try:
        built = builder(artifacts)
    except TypeError:
        # Builder may accept no args in an alternate implementation.
        try:
            built = builder()
        except Exception as exc:  # noqa: BLE001
            logger.warning("charts.build_all_charts failed (%s); omitting figures.", exc)
            return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("charts.build_all_charts failed (%s); omitting figures.", exc)
        return []

    return _normalise_chart_paths(built)


def _normalise_chart_paths(built: Any) -> List[Tuple[str, Path]]:
    """Coerce a variety of return shapes into ``[(label, Path), ...]``.

    Accepts: dict[label->path], dict[label->{'path':...}], list[path|dict],
    or a single path. Only existing ``.png``/``.jpg`` files are kept.
    """
    out: List[Tuple[str, Path]] = []

    def add(label: Any, value: Any) -> None:
        path: Optional[Path] = None
        if isinstance(value, (str, Path)):
            path = Path(value)
        elif isinstance(value, dict):
            for key in ("path", "file", "png", "filepath"):
                if value.get(key):
                    path = Path(value[key])
                    break
            label = value.get("title") or value.get("label") or label
        if path is None:
            return
        try:
            if path.exists() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                out.append((_prettify_label(str(label)), path))
        except OSError:
            pass

    try:
        if isinstance(built, dict):
            for k, v in built.items():
                add(k, v)
        elif isinstance(built, (list, tuple, set)):
            for idx, v in enumerate(built):
                add(f"Figure {idx + 1}", v)
        elif built is not None:
            add("Figure", built)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not normalise chart paths (%s).", exc)
    return out


def _prettify_label(name: str) -> str:
    base = Path(name).stem if ("/" in name or "\\" in name) else name
    base = re.sub(r"[_\-]+", " ", base).strip()
    return base[:1].upper() + base[1:] if base else "Figure"


# ─────────────────────────────────────────────────────────────────────────────
# Metrics table rows (from eval artifact)
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_rows(ev: Optional[Dict[str, Any]]) -> List[List[str]]:
    """Flatten the eval payload into ``[[Metric, Value], ...]`` display rows."""
    rows: List[List[str]] = [["Metric", "Value"]]
    if not ev:
        rows.append(["(no eval artifact found)", "run `kbqa evaluate`"])
        return rows

    k = ev.get("k", 5)
    n_q = ev.get("n_questions")
    retr = ev.get("retrieval", {}) or {}
    ans = ev.get("answer", {}) or {}
    abst = ev.get("abstention", {}) or {}

    def pct(x: Any) -> str:
        try:
            v = float(x)
            return f"{v * 100:.1f}%" if v <= 1.0 else f"{v:.1f}"
        except (TypeError, ValueError):
            return str(x)

    if n_q is not None:
        rows.append(["Questions evaluated", str(n_q)])
    hy = retr.get(f"recall@{k}_hybrid")
    bm = retr.get(f"recall@{k}_bm25")
    if hy is None:  # tolerate alternate key spellings
        hy = next((v for kk, v in retr.items() if "hybrid" in str(kk)), None)
    if bm is None:
        bm = next((v for kk, v in retr.items() if "bm25" in str(kk)), None)
    if hy is not None:
        rows.append([f"Recall@{k} (hybrid)", pct(hy)])
    if bm is not None:
        rows.append([f"Recall@{k} (BM25)", pct(bm)])
    if ans.get("exact_match") is not None:
        rows.append(["Answer Exact Match", f"{float(ans['exact_match']):.1f}"])
    if ans.get("f1") is not None:
        rows.append(["Answer F1", f"{float(ans['f1']):.1f}"])
    if ans.get("n_answerable") is not None:
        rows.append(["Answerable questions", str(ans["n_answerable"])])
    if abst.get("abstain_recall") is not None:
        rows.append(["Abstention recall", pct(abst["abstain_recall"])])
    if abst.get("n_unanswerable") is not None:
        rows.append(["Unanswerable questions", str(abst["n_unanswerable"])])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Output path
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_out_path(out_path, stamp: str, ext: str = ".pdf") -> Path:
    """Resolve the destination path, creating the submission directory."""
    if out_path is not None:
        p = Path(out_path)
        if p.suffix.lower() != ext:
            p = p / f"report{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    sub = artifacts_dir() / "submission" / f"submission-{stamp}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"report{ext}"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown fallback (no reportlab)
# ─────────────────────────────────────────────────────────────────────────────

def _write_md_fallback(title: str, author: str, out_path: Path,
                       ev: Optional[Dict[str, Any]],
                       charts: Sequence[Tuple[str, Path]]) -> Path:
    """Write a Markdown digest when ReportLab is unavailable."""
    md_path = out_path.with_suffix(".md")
    parts: List[str] = []
    parts.append(f"# {title}\n")
    parts.append(f"**Author:** {author}  \n**Generated:** {utc_now_iso()}\n")
    parts.append("\n> ReportLab is not installed; this Markdown digest is a fallback "
                 "for the PDF report. `pip install reportlab` to produce the PDF.\n")

    # Metrics
    parts.append("\n## Evaluation Summary\n")
    for row in _metrics_rows(ev):
        parts.append(f"- **{row[0]}:** {row[1]}")
    if charts:
        parts.append("\n## Figures\n")
        for label, p in charts:
            parts.append(f"- {label}: `{p}`")

    docs = _docs_dir()
    for stem, heading in _DOC_SECTIONS:
        f = docs / f"{stem}.md"
        parts.append(f"\n\n---\n\n## {heading}\n")
        if f.exists():
            try:
                parts.append(f.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                parts.append(f"_(could not read {f.name}: {exc})_")
        else:
            parts.append(f"_(missing source: {f.name})_")

    md_path.write_text("\n".join(parts), encoding="utf-8")
    logger.warning("ReportLab missing — wrote Markdown fallback -> %s", md_path)
    return md_path


# ─────────────────────────────────────────────────────────────────────────────
# PDF construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_styles():
    """Construct the paragraph stylesheet (reportlab imported by caller)."""
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="KBTitle", parent=styles["Title"], fontSize=26, leading=32,
        spaceAfter=18, alignment=TA_CENTER))
    styles.add(ParagraphStyle(
        name="KBSubtitle", parent=styles["Normal"], fontSize=13, leading=18,
        alignment=TA_CENTER, textColor=_grey()))
    styles.add(ParagraphStyle(
        name="KBH1", parent=styles["Heading1"], fontSize=17, leading=21,
        spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(
        name="KBH2", parent=styles["Heading2"], fontSize=13.5, leading=17,
        spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle(
        name="KBH3", parent=styles["Heading3"], fontSize=11.5, leading=15,
        spaceBefore=7, spaceAfter=3))
    styles.add(ParagraphStyle(
        name="KBBody", parent=styles["BodyText"], fontSize=9.5, leading=13.5,
        alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle(
        name="KBBullet", parent=styles["BodyText"], fontSize=9.5, leading=13,
        leftIndent=14, bulletIndent=4, spaceAfter=2))
    styles.add(ParagraphStyle(
        name="KBCode", parent=styles["Code"], fontSize=7.6, leading=9.6,
        backColor=_lightgrey(), borderPadding=4, leftIndent=4))
    styles.add(ParagraphStyle(
        name="KBQuote", parent=styles["BodyText"], fontSize=9.5, leading=13,
        leftIndent=16, textColor=_grey(), fontName="Helvetica-Oblique"))
    styles.add(ParagraphStyle(
        name="KBCell", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(
        name="KBCellHead", parent=styles["BodyText"], fontSize=8, leading=10,
        textColor=_white(), fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(
        name="KBCaption", parent=styles["Normal"], fontSize=8.5, leading=11,
        alignment=TA_CENTER, textColor=_grey(), spaceBefore=3, spaceAfter=10))
    return styles


def _grey():
    from reportlab.lib import colors
    return colors.HexColor("#555555")


def _white():
    from reportlab.lib import colors
    return colors.white


def _lightgrey():
    from reportlab.lib import colors
    return colors.HexColor("#f2f2f2")


def _accent():
    from reportlab.lib import colors
    return colors.HexColor("#1f4e79")


def _blocks_to_flowables(blocks: Sequence[Dict[str, Any]], styles) -> List[Any]:
    """Convert parsed markdown blocks into ReportLab flowables for one section."""
    from reportlab.lib.units import mm
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, Spacer

    flows: List[Any] = []
    for b in blocks:
        kind = b.get("kind")
        try:
            if kind == "heading":
                level = int(b.get("level", 2))
                style = {2: "KBH2", 3: "KBH3"}.get(level, "KBH3" if level >= 4 else "KBH1")
                flows.append(Paragraph(_strip_inline(b.get("text", "")), styles[style]))
            elif kind == "para":
                flows.append(Paragraph(_strip_inline(b["text"]), styles["KBBody"]))
            elif kind in ("ulist", "olist"):
                items = [ListItem(Paragraph(_strip_inline(it), styles["KBBullet"]),
                                  leftIndent=14)
                         for it in b.get("items", []) if it.strip()]
                if items:
                    flows.append(ListFlowable(
                        items, bulletType="1" if kind == "olist" else "bullet",
                        start="1" if kind == "olist" else None,
                        bulletFontSize=8, leftIndent=10))
                    flows.append(Spacer(1, 3))
            elif kind == "table":
                tbl = _make_table(b.get("rows", []), styles)
                if tbl is not None:
                    flows.append(Spacer(1, 2))
                    flows.append(tbl)
                    flows.append(Spacer(1, 5))
            elif kind == "code":
                flows.append(_code_flowable(b.get("text", ""), styles))
            elif kind == "quote":
                flows.append(Paragraph(_strip_inline(b.get("text", "")), styles["KBQuote"]))
                flows.append(Spacer(1, 4))
            elif kind == "hr":
                flows.append(Spacer(1, 4 * mm))
        except Exception as exc:  # noqa: BLE001 — skip any pathological block
            logger.debug("Skipping a %s block: %s", kind, exc)
    return flows


def _code_flowable(code: str, styles):
    """Render a code block as a wrapped, escaped Preformatted-ish paragraph."""
    from reportlab.platypus import Paragraph
    # Cap absurdly long code blocks so they don't dominate the page.
    lines = code.split("\n")
    if len(lines) > 28:
        lines = lines[:28] + ["...  (truncated)"]
    esc = "<br/>".join(html.escape(ln, quote=False).replace(" ", "&nbsp;") for ln in lines)
    return Paragraph(esc, styles["KBCode"])


def _make_table(rows: Sequence[Sequence[str]], styles, header: bool = True):
    """Build a styled ReportLab Table from string rows (best-effort)."""
    if not rows:
        return None
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Table, TableStyle

    ncols = max(len(r) for r in rows)
    if ncols == 0:
        return None
    # Available width on A4 with our margins (~170mm).
    avail = 170 * mm
    col_w = avail / ncols

    data: List[List[Any]] = []
    for ri, row in enumerate(rows):
        cells = list(row) + [""] * (ncols - len(row))
        style = "KBCellHead" if (header and ri == 0) else "KBCell"
        data.append([Paragraph(_strip_inline(str(c)), styles[style]) for c in cells])

    tbl = Table(data, colWidths=[col_w] * ncols, repeatRows=1 if header else 0)
    ts = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fb")]),
    ]
    if header:
        ts.append(("BACKGROUND", (0, 0), (-1, 0), _accent()))
    tbl.setStyle(TableStyle(ts))
    return tbl


def _title_page(title: str, author: str, styles, metrics_rows) -> List[Any]:
    """Build the title page flowables."""
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, Spacer

    flows: List[Any] = [Spacer(1, 55 * mm)]
    flows.append(Paragraph(_strip_inline(title), styles["KBTitle"]))
    flows.append(Spacer(1, 6 * mm))
    flows.append(Paragraph("RAG-based Knowledge Base Question Answering", styles["KBSubtitle"]))
    flows.append(Spacer(1, 18 * mm))
    flows.append(Paragraph(_strip_inline(author), styles["KBSubtitle"]))
    date_str = utc_now_iso().split("T")[0]
    flows.append(Paragraph(f"Generated {date_str}", styles["KBSubtitle"]))
    flows.append(Spacer(1, 24 * mm))
    # A compact results table on the title page if we have metrics.
    if metrics_rows and len(metrics_rows) > 1:
        flows.append(Paragraph("Evaluation Snapshot", styles["KBSubtitle"]))
        flows.append(Spacer(1, 4 * mm))
        tbl = _make_table(metrics_rows, styles)
        if tbl is not None:
            tbl.hAlign = "CENTER"
            flows.append(tbl)
    flows.append(PageBreak())
    return flows


def _overview_section(styles, ev, n_charts: int) -> List[Any]:
    """A short executive-overview section right after the title page."""
    from reportlab.platypus import Paragraph, Spacer

    flows: List[Any] = [Paragraph("Executive Overview", styles["KBH1"])]
    summary = (
        "This report documents an agentic Retrieval-Augmented Generation (RAG) system "
        "for question answering over a private document knowledge base. The pipeline "
        "ingests and chunks documents, indexes them in FAISS, then for each query runs "
        "query analysis, hybrid retrieval (BM25 + dense), cross-encoder reranking, a "
        "sufficiency check, grounded answer generation with citations, and a "
        "faithfulness self-check. The agent prefers an explicit abstention over an "
        "ungrounded guess. The sections that follow are assembled from the project's "
        "design documents and are accompanied by evaluation figures and metrics drawn "
        "from the latest run artifacts."
    )
    flows.append(Paragraph(summary, styles["KBBody"]))
    flows.append(Spacer(1, 4))
    if ev:
        flows.append(Paragraph("Key Results", styles["KBH2"]))
        tbl = _make_table(_metrics_rows(ev), styles)
        if tbl is not None:
            flows.append(tbl)
    if n_charts:
        flows.append(Spacer(1, 4))
        flows.append(Paragraph(
            f"{n_charts} evaluation figure(s) are included in the Figures section.",
            styles["KBBody"]))
    return flows


def _charts_section(charts: Sequence[Tuple[str, Path]], styles) -> List[Any]:
    """Embed chart PNGs, each scaled to fit the page width."""
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, Spacer

    flows: List[Any] = [PageBreak(), Paragraph("Figures", styles["KBH1"])]
    max_w = 165 * mm
    max_h = 105 * mm
    for label, path in charts:
        try:
            iw, ih = ImageReader(str(path)).getSize()
            if not iw or not ih:
                raise ValueError("zero image size")
            scale = min(max_w / iw, max_h / ih)
            w, h = iw * scale, ih * scale
            img = Image(str(path), width=w, height=h)
            cap = Paragraph(label, styles["KBCaption"])
            flows.append(KeepTogether([Spacer(1, 4), img, cap]))
        except Exception as exc:  # noqa: BLE001 — skip unreadable figure
            logger.warning("Skipping chart %s (%s).", path, exc)
    return flows


def _on_page(canvas, doc):
    """Page footer with page number + author/version, drawn on every page."""
    try:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        from reportlab.lib import colors
        canvas.setFillColor(colors.HexColor("#888888"))
        w, _h = canvas._pagesize
        canvas.drawCentredString(w / 2.0, 12, f"Page {doc.page}")
        canvas.restoreState()
    except Exception:  # noqa: BLE001
        pass


def generate_report(cfg: AppConfig, title: str = None, author: str = None,
                    out_path=None) -> Path:
    """Generate the submission PDF report and return its path.

    Parameters
    ----------
    cfg : AppConfig
        Active configuration (provides default title/author and path roots).
    title, author : str, optional
        Override the report title/author; default to ``cfg.project_title`` and
        ``cfg.author``.
    out_path : str | Path, optional
        Explicit destination. A directory or extension-less path is treated as a
        folder and ``report.pdf`` is written inside it. Defaults to
        ``artifacts_dir()/submission/submission-<stamp>/report.pdf``.

    Returns
    -------
    pathlib.Path
        Path to the produced ``report.pdf`` — or ``report.md`` if ReportLab is
        unavailable.
    """
    title = title or getattr(cfg, "project_title", "Knowledge Base QA")
    author = author or getattr(cfg, "author", "")
    stamp = utc_stamp()

    # Gather data sources up front (all defensive / optional).
    ev = _latest_eval()
    charts = _collect_charts()

    pdf_path = _resolve_out_path(out_path, stamp, ".pdf")

    # ── Lazy reportlab import; fall back to Markdown on failure ───────────────
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            PageBreak, Paragraph, SimpleDocTemplate, Spacer,
        )
    except Exception as exc:  # noqa: BLE001 — reportlab missing or broken
        logger.warning("ReportLab unavailable (%s); writing Markdown fallback.", exc)
        return _write_md_fallback(title, author, pdf_path, ev, charts)

    try:
        styles = _build_styles()
        metrics_rows = _metrics_rows(ev)

        story: List[Any] = []
        # 1) Title page
        story += _title_page(title, author, styles, metrics_rows)
        # 2) Executive overview + key results
        story += _overview_section(styles, ev, len(charts))

        # 3) Narrative sections from docs/
        docs = _docs_dir()
        for stem, heading in _DOC_SECTIONS:
            src = docs / f"{stem}.md"
            story.append(PageBreak())
            story.append(Paragraph(_strip_inline(heading), styles["KBH1"]))
            if not src.exists():
                story.append(Paragraph(
                    f"<i>Source document not found: {src.name}.</i>", styles["KBBody"]))
                logger.warning("Doc missing for report: %s", src)
                continue
            try:
                text = src.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                story.append(Paragraph(
                    f"<i>Could not read {src.name}: {exc}.</i>", styles["KBBody"]))
                continue
            blocks = _parse_markdown(text)
            # Drop the leading H1 (it duplicates our section heading).
            if blocks and blocks[0].get("kind") == "heading" and blocks[0].get("level") == 1:
                blocks = blocks[1:]
            story += _blocks_to_flowables(blocks, styles)

        # 4) Figures appendix
        if charts:
            story += _charts_section(charts, styles)

        # Build the document.
        version = getattr(getattr(cfg, "serving", None), "model_version", "v1")
        doc = SimpleDocTemplate(
            str(pdf_path), pagesize=A4,
            leftMargin=20 * mm, rightMargin=20 * mm,
            topMargin=18 * mm, bottomMargin=18 * mm,
            title=title, author=author or "KBQA",
            subject="RAG Knowledge Base QA — Project Report",
        )
        doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
        logger.info("Report -> %s (%d figures, model_version=%s)",
                    pdf_path, len(charts), version)
        return pdf_path
    except Exception as exc:  # noqa: BLE001 — never crash the autopilot
        logger.exception("PDF build failed (%s); writing Markdown fallback.", exc)
        return _write_md_fallback(title, author, pdf_path, ev, charts)
