"""Agent state + audit primitives for the RAG QA agent.

``AgentState`` threads through every step (analyze → retrieve → rerank →
sufficiency → generate → faithfulness). It is fully serialisable so the API can
return the complete reasoning trace + the citations behind each answer — the
transparency that makes a RAG answer trustworthy and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class AnswerStatus(str, Enum):
    ANSWERED = "answered"            # grounded answer produced
    INSUFFICIENT = "insufficient"   # retrieved context not enough → abstain
    CLARIFY = "needs_clarification"  # ambiguous question → ask back
    NO_ANSWER = "no_answer"         # answerable form but KB has no answer


@dataclass
class RetrievedPassage:
    id: str
    text: str
    title: str = ""
    score: float = 0.0
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    summary: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    # --- input ---
    question: str
    query_id: str = "q"

    # --- intermediate ---
    sub_queries: List[str] = field(default_factory=list)
    is_multi_hop: bool = False
    retrieved: List[RetrievedPassage] = field(default_factory=list)
    reranked: List[RetrievedPassage] = field(default_factory=list)
    context_sufficient: bool = False
    retrieval_attempts: int = 0

    # --- output ---
    status: Optional[AnswerStatus] = None
    answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    faithfulness: float = 0.0
    clarifying_questions: List[str] = field(default_factory=list)
    rationale: str = ""

    # --- audit ---
    trace: List[ToolTrace] = field(default_factory=list)
    model_versions: Dict[str, str] = field(default_factory=dict)

    def add_trace(self, t: ToolTrace) -> None:
        self.trace.append(t)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, AnswerStatus) else self.status
        return d


__all__ = ["AnswerStatus", "RetrievedPassage", "ToolTrace", "AgentState"]
