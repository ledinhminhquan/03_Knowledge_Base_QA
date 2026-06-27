"""FastAPI service for the Knowledge Base QA system.

Endpoints: GET /health, POST /ingest, POST /search, POST /ask, POST /batch,
GET /metrics. Models load once into a singleton agent; every response echoes
``model_versions`` for traceability.
"""

from __future__ import annotations

import time
from typing import Dict

from ..logging_utils import get_logger
from .dependencies import get_config, get_rag_agent
from .schemas import (AskRequest, AskResponse, BatchAskRequest, BatchAskResponse,
                      HealthResponse, IngestRequest, IngestResponse, PassageOut,
                      SearchRequest, SearchResponse)

logger = get_logger(__name__)


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse

    cfg = get_config()
    app = FastAPI(title=cfg.serving.api_title, version=cfg.serving.api_version)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    try:
        from prometheus_client import Counter, Histogram, generate_latest
        REQ = Counter("kbqa_requests_total", "Requests", ["endpoint"])
        LAT = Histogram("kbqa_latency_seconds", "Latency", ["endpoint"])
        _PROM = True
    except Exception:
        REQ = LAT = None
        _PROM = False

    def _obs(ep, sec):
        if _PROM:
            REQ.labels(ep).inc(); LAT.labels(ep).observe(sec)

    @app.get("/health", response_model=HealthResponse)
    def health():
        agent = get_rag_agent()
        return HealthResponse(
            status="ok", version=cfg.serving.api_version, model_version=cfg.serving.model_version,
            index={"loaded": agent.retriever.size > 0, "n_vectors": agent.retriever.size},
            model_versions={"retrieve": agent.retrieve_tool.version, "rerank": agent.rerank_tool.version,
                            "generate": agent.generate_tool.version, "orchestrator": agent.orchestrator.name},
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest):
        t0 = time.perf_counter()
        agent = get_rag_agent()
        docs = [d.model_dump() for d in req.documents]
        out = agent.ingest(docs)
        dt = time.perf_counter() - t0
        _obs("ingest", dt)
        return IngestResponse(**out, model_version=cfg.serving.model_version, took_ms=round(dt * 1000, 2))

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        t0 = time.perf_counter()
        agent = get_rag_agent()
        cands = agent.retrieve_tool.run(query=req.query, top_k=req.top_k).get("passages", [])
        reranked = agent.rerank_tool.run(query=req.query, candidates=cands, top_n=req.rerank_top_n).get("passages", [])
        dt = time.perf_counter() - t0
        _obs("search", dt)
        passages = [PassageOut(chunk_id=p.get("id", ""), doc_id=p.get("doc_id", ""), title=p.get("title", ""),
                               text=p.get("text", ""), retriever_score=p.get("score", 0.0),
                               rerank_score=p.get("rerank_score"), rank=p.get("rank", i))
                    for i, p in enumerate(reranked)]
        return SearchResponse(passages=passages, timing_ms=round(dt * 1000, 2), model_version=cfg.serving.model_version)

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        t0 = time.perf_counter()
        agent = get_rag_agent()
        state = agent.ask(req.question)
        dt = time.perf_counter() - t0
        _obs("ask", dt)
        d = state.to_dict()
        return AskResponse(
            question=req.question, answer=d.get("answer", ""), status=d.get("status"),
            is_answerable=d.get("status") == "answered", citations=d.get("citations", []),
            confidence=d.get("confidence", 0.0), faithfulness=d.get("faithfulness", 0.0),
            is_multi_hop=d.get("is_multi_hop", False), clarifying_questions=d.get("clarifying_questions", []),
            trace=d.get("trace", []) if req.return_trace else [], model_versions=d.get("model_versions", {}),
            timing_ms=round(dt * 1000, 2),
        )

    @app.post("/batch", response_model=BatchAskResponse)
    def batch(req: BatchAskRequest):
        t0 = time.perf_counter()
        if len(req.questions) > cfg.serving.max_batch_questions:
            raise HTTPException(status_code=413, detail=f"Too many questions (max {cfg.serving.max_batch_questions}).")
        agent = get_rag_agent()
        results = []
        for q in req.questions:
            state = agent.ask(q)
            d = state.to_dict()
            results.append(AskResponse(question=q, answer=d.get("answer", ""), status=d.get("status"),
                                       is_answerable=d.get("status") == "answered", citations=d.get("citations", []),
                                       confidence=d.get("confidence", 0.0), faithfulness=d.get("faithfulness", 0.0),
                                       model_versions=d.get("model_versions", {})))
        dt = time.perf_counter() - t0
        return BatchAskResponse(results=results, count=len(results), took_ms=round(dt * 1000, 2))

    @app.get("/metrics")
    def metrics():
        if not _PROM:
            return PlainTextResponse("prometheus_client not installed", status_code=501)
        from prometheus_client import generate_latest
        return PlainTextResponse(generate_latest().decode("utf-8"))

    return app


app = create_app()

__all__ = ["app", "create_app"]
