"""Text cleaning / normalisation shared by ingestion, training and inference."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

_MULTISPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_whitespace(text: str) -> str:
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def clean_text(text: Optional[str], max_chars: int = 100000) -> str:
    """NFKC-normalise, strip control chars and collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub(" ", text)
    text = normalize_whitespace(text)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def normalize_answer(s: str) -> str:
    """SQuAD-style answer normalisation for EM/F1 (lowercase, strip punct/articles)."""
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


__all__ = ["clean_text", "normalize_whitespace", "normalize_answer"]
