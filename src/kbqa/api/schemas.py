"""Pydantic request/response schemas for the KBQA REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    text: str
    id: Optional[str] = None
    title: str = ""
    source: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: List[Document]


class IngestResponse(BaseModel):
    ingested_docs: int
    new_chunks: int
    index_n_vectors: int
    model_version: str = "v1"
    took_ms: float = 0.0


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(20, ge=1, le=200)
    rerank_top_n: int = Field(5, ge=1, le=50)


class PassageOut(BaseModel):
    chunk_id: str
    doc_id: str = ""
    title: str = ""
    text: str
    retriever_score: float = 0.0
    rerank_score: Optional[float] = None
    rank: int = 0


class SearchResponse(BaseModel):
    passages: List[PassageOut] = Field(default_factory=list)
    timing_ms: float = 0.0
    model_version: str = "v1"


class AskRequest(BaseModel):
    question: str
    return_trace: bool = True


class Citation(BaseModel):
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    title: Optional[str] = None
    quote: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    status: Optional[str] = None
    is_answerable: bool = True
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    faithfulness: float = 0.0
    is_multi_hop: bool = False
    clarifying_questions: List[str] = Field(default_factory=list)
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    model_versions: Dict[str, str] = Field(default_factory=dict)
    timing_ms: float = 0.0


class BatchAskRequest(BaseModel):
    questions: List[str]


class BatchAskResponse(BaseModel):
    results: List[AskResponse] = Field(default_factory=list)
    count: int = 0
    took_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    model_version: str = "v1"
    index: Dict[str, Any] = Field(default_factory=dict)
    model_versions: Dict[str, str] = Field(default_factory=dict)


__all__ = ["Document", "IngestRequest", "IngestResponse", "SearchRequest", "PassageOut",
           "SearchResponse", "AskRequest", "Citation", "AskResponse", "BatchAskRequest",
           "BatchAskResponse", "HealthResponse"]
