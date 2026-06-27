"""Agent + retrieval tests that run CPU-only with NO model downloads.

We build a BM25-only retriever directly (no dense encoder) and disable the
reranker + reader, proving the graceful-degradation contract: the full RAG agent
runs end-to-end on a clean machine and abstains safely when no reader is present.
"""
import pytest

from kbqa.config import AppConfig
from kbqa.models.baseline_bm25 import BM25Retriever
from kbqa.data.chunking import chunk_corpus
from kbqa.data.samples import SAMPLE_DOCS
from kbqa.agent.rag_agent import RAGAgent
from kbqa.agent.state import AnswerStatus


def test_bm25_finds_relevant_passage():
    passages = chunk_corpus(SAMPLE_DOCS, chunk_size=120, overlap=20)
    bm25 = BM25Retriever().index(passages)
    hits = bm25.search("What does FAISS stand for?", top_k=3)
    assert hits
    assert any("FAISS" in p.text or "Similarity Search" in p.text for p, _ in hits)


@pytest.fixture()
def bm25_agent():
    cfg = AppConfig()
    cfg.serving.log_queries = False
    agent = RAGAgent(cfg, load_kb=False, load_reranker=False, load_reader=False)
    # attach a BM25-only retriever (no dense encoder → no downloads)
    agent.retriever.store = None
    agent.retriever.bm25 = BM25Retriever().index(chunk_corpus(SAMPLE_DOCS, chunk_size=120, overlap=20))
    agent.retrieve_tool.retriever = agent.retriever
    return agent


def test_agent_retrieves_and_traces(bm25_agent):
    state = bm25_agent.ask("What does FAISS stand for?")
    d = state.to_dict()
    assert d["status"] in {s.value for s in AnswerStatus}
    assert d["retrieved"], "BM25 should retrieve passages"
    tools = {t["tool"] for t in d["trace"]}
    assert "retrieve" in tools and "sufficiency" in tools


def test_agent_abstains_without_reader(bm25_agent):
    # No reader is loaded → the agent must abstain rather than fabricate.
    state = bm25_agent.ask("What does FAISS stand for?")
    assert state.status in {AnswerStatus.INSUFFICIENT, AnswerStatus.NO_ANSWER}
    assert "don't have enough information" in state.answer.lower() or state.answer == "I don't know."


def test_unanswerable_question_routes_out(bm25_agent):
    state = bm25_agent.ask("Do you think the stock market will go up tomorrow?")
    assert state.status in {AnswerStatus.NO_ANSWER, AnswerStatus.INSUFFICIENT}
