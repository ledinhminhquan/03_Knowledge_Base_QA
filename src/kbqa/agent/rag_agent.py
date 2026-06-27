"""The agentic RAG pipeline.

Wires the tools + orchestrator into the control loop:

    analyze → (per query) [retrieve → rerank → sufficiency]×(CRAG loop)
            → generate (grounded) → faithfulness gate → answer + citations | abstain

Models load once (singletons) with graceful fallback at every layer, so the agent
runs end-to-end even with only the built-in sample KB and the rule brain.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..config import AppConfig, index_dir
from ..logging_utils import JsonlLogger, get_logger
from . import policy
from .llm_orchestrator import make_orchestrator
from .state import AgentState, AnswerStatus, RetrievedPassage, ToolTrace
from .tools import GenerateTool, RerankTool, RetrieveTool

logger = get_logger(__name__)


class RAGAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, *, load_kb: bool = True,
                 load_reranker: bool = True, load_reader: bool = True):
        self.cfg = cfg or AppConfig()
        self.orchestrator = make_orchestrator(self.cfg.agent)

        self.retriever = self._load_retriever(load_kb)
        reranker = self._load_reranker(load_reranker)
        reader = self._load_reader(load_reader)

        self.retrieve_tool = RetrieveTool(self.retriever)
        self.rerank_tool = RerankTool(reranker)
        self.generate_tool = GenerateTool(reader, mode=self.cfg.agent.reader_mode)

        self._qlog = JsonlLogger(self.cfg.serving.query_log_path) if self.cfg.serving.log_queries else None

    # ---- loading -----------------------------------------------------------
    def _load_retriever(self, load_kb: bool):
        from ..models.retriever import Retriever
        retriever = Retriever(self.cfg.retriever, model_dir=str(self.cfg.retriever.output_dir))
        if load_kb and (Path(index_dir()) / "store_meta.json").exists():
            try:
                retriever.load(str(index_dir()))
            except Exception as exc:
                logger.warning("Could not load KB index (%s); KB empty until /ingest.", exc)
        return retriever

    def _load_reranker(self, load_reranker: bool):
        if not load_reranker:
            return None
        try:
            from ..models.reranker import Reranker
            return Reranker(self.cfg.reranker)
        except Exception as exc:
            logger.warning("Reranker unavailable (%s).", exc)
            return None

    def _load_reader(self, load_reader: bool):
        if not load_reader:
            return None
        try:
            if self.cfg.agent.reader_mode == "generative":
                from ..models.reader_generative import GenerativeReader
                return GenerativeReader(self.cfg.generator, model_dir=str(self.cfg.generator.output_dir))
            from ..models.reader_extractive import ExtractiveReader
            return ExtractiveReader(self.cfg.reader, model_dir=str(self.cfg.reader.output_dir))
        except Exception as exc:
            logger.warning("Reader unavailable (%s).", exc)
            return None

    # ---- ingestion ---------------------------------------------------------
    def ingest(self, documents: Sequence[Dict]) -> Dict:
        from ..data.chunking import chunk_corpus
        passages = chunk_corpus(documents, chunk_size=self.cfg.chunk.chunk_size_words,
                                overlap=self.cfg.chunk.overlap_words)
        before = self.retriever.size
        self.retriever.add(passages)
        after = self.retriever.size
        self.retrieve_tool.retriever = self.retriever
        return {"ingested_docs": len(documents), "new_chunks": after - before, "index_n_vectors": after}

    # ---- helpers -----------------------------------------------------------
    def _trace(self, state: AgentState, tool, out: Dict, summary: str) -> None:
        state.add_trace(ToolTrace(tool=tool.name if hasattr(tool, "name") else str(tool),
                                  ok=out.get("ok", True), latency_ms=out.get("latency_ms", 0.0),
                                  summary=summary, error=out.get("error")))
        if hasattr(tool, "version") and out.get("ok", True):
            state.model_versions[tool.name] = tool.version

    def _crag_loop(self, query: str, state: AgentState) -> List[Dict]:
        """Retrieve → rerank → sufficiency, widening on AMBIGUOUS (CRAG)."""
        reranked: List[Dict] = []
        top_k = self.cfg.retriever.top_k
        for it in range(self.cfg.agent.max_iterations):
            state.retrieval_attempts += 1
            q = query if it == 0 else policy.expand_query(query)
            r_out = self.retrieve_tool.run(query=q, top_k=top_k)
            cands = r_out.get("passages", [])
            self._trace(state, self.retrieve_tool, r_out, f"iter{it}: {len(cands)} retrieved (k={top_k})")

            rr_out = self.rerank_tool.run(query=query, candidates=cands, top_n=self.cfg.reranker.rerank_top_n)
            reranked = rr_out.get("passages", [])
            self._trace(state, self.rerank_tool, rr_out, f"iter{it}: reranked -> {len(reranked)}")

            suff = self.orchestrator.sufficiency(query, reranked, self.cfg.agent)
            state.add_trace(ToolTrace(tool="sufficiency", ok=True, latency_ms=0.0,
                                      summary=f"iter{it}: {suff.get('verdict')} (score={suff.get('score')})"))
            state.context_sufficient = suff.get("verdict") == "SUFFICIENT"
            if suff.get("verdict") == "SUFFICIENT":
                break
            top_k = min(top_k * 2, 100)  # widen on retry
        return reranked

    # ---- main entrypoint ---------------------------------------------------
    def ask(self, question: str, query_id: str = "q") -> AgentState:
        state = AgentState(question=question, query_id=query_id)

        analysis = self.orchestrator.analyze(question, self.cfg.agent)
        state.is_multi_hop = analysis.get("is_multi_hop", False)
        state.sub_queries = analysis.get("sub_questions", [])
        state.add_trace(ToolTrace(tool="analyze_query", ok=True, latency_ms=0.0,
                                  summary=f"qtype={analysis.get('qtype')} multihop={state.is_multi_hop}"))

        if analysis.get("qtype") == "unanswerable":
            return self._finish(state, AnswerStatus.NO_ANSWER, "I don't know.",
                                rationale="Question is subjective/unanswerable from a knowledge base.")

        queries = state.sub_queries if (state.is_multi_hop and state.sub_queries) else [analysis.get("rewritten", question)]

        merged: Dict[str, Dict] = {}
        for q in queries:
            for p in self._crag_loop(q, state):
                merged.setdefault(p["id"], p)
        reranked = sorted(merged.values(), key=lambda c: c.get("rerank_score", c.get("score", 0)), reverse=True)
        reranked = reranked[: self.cfg.reranker.rerank_top_n]
        state.retrieved = [_rp(p) for p in reranked]
        state.reranked = state.retrieved

        if not reranked:
            return self._finish(state, AnswerStatus.INSUFFICIENT,
                                "I don't have enough information in the knowledge base.",
                                rationale="No passages retrieved.")

        gen = self.generate_tool.run(question=question, passages=reranked)
        self._trace(state, self.generate_tool, gen, f"answer='{(gen.get('answer') or '')[:40]}' abstain={gen.get('abstain')}")
        answer = (gen.get("answer") or "").strip()

        if gen.get("abstain") or not answer:
            return self._finish(state, AnswerStatus.INSUFFICIENT,
                                "I don't have enough information in the knowledge base.",
                                rationale="Reader abstained (no supported span).")

        faith = self.orchestrator.faithfulness(answer, reranked, self.cfg.agent, encoder=getattr(self.retriever, "encoder", None))
        state.faithfulness = faith.get("support_score", 0.0)
        state.add_trace(ToolTrace(tool="faithfulness", ok=True, latency_ms=0.0,
                                  summary=f"supported={faith.get('supported')} score={state.faithfulness}"))
        if self.cfg.agent.require_citations and not faith.get("supported", False):
            return self._finish(state, AnswerStatus.INSUFFICIENT,
                                "I don't have enough information in the knowledge base.",
                                rationale="Answer not grounded in retrieved context (faithfulness gate).")

        state.citations = gen.get("citations", [])
        top_rr = reranked[0].get("rerank_score", reranked[0].get("score", 0.0))
        state.confidence = round(min(1.0, 0.5 * float(top_rr) + 0.5 * state.faithfulness), 3)
        return self._finish(state, AnswerStatus.ANSWERED, answer,
                            rationale="Grounded answer extracted from retrieved context.")

    def _finish(self, state: AgentState, status: AnswerStatus, answer: str, rationale: str = "") -> AgentState:
        state.status = status
        state.answer = answer
        state.rationale = rationale
        if status in (AnswerStatus.INSUFFICIENT, AnswerStatus.NO_ANSWER) and not state.clarifying_questions:
            missing = []
            state.clarifying_questions = [self.orchestrator.clarify(state.question, missing)]
        state.model_versions["orchestrator"] = self.orchestrator.name
        if self._qlog is not None:
            try:
                self._qlog.log("ask", query_id=state.query_id, status=str(status.value),
                               confidence=state.confidence, faithfulness=state.faithfulness,
                               n_citations=len(state.citations), multihop=state.is_multi_hop)
            except Exception:
                pass
        return state


def _rp(p: Dict) -> RetrievedPassage:
    return RetrievedPassage(id=p.get("id", ""), text=p.get("text", ""), title=p.get("title", ""),
                            score=float(p.get("score", 0.0)), rerank_score=p.get("rerank_score"))


_AGENT: Optional[RAGAgent] = None


def get_agent(cfg: Optional[AppConfig] = None, **kwargs) -> RAGAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = RAGAgent(cfg, **kwargs)
    return _AGENT


__all__ = ["RAGAgent", "get_agent"]
