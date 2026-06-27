"""Knowledge Base Question-Answering System (RAG).

A production, agentic Retrieval-Augmented Generation system that answers natural
language questions over a document knowledge base: ingest → chunk → index (FAISS)
→ query analysis → retrieve → rerank → sufficiency check → grounded generation
with citations → faithfulness self-check.

Runs fully on CPU with open models and a deterministic agent; upgrades to a
fine-tuned retriever/reader and an optional LLM brain when available.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
