"""Dataset loading for training + the demo knowledge base.

Loads the verified public datasets:
  * reader        : ``rajpurkar/squad_v2``                  (extractive QA + abstain)
  * retriever     : ``sentence-transformers/natural-questions`` (query/answer pairs)
  * demo KB       : ``rag-datasets/rag-mini-wikipedia``      (corpus + QA)
  * recall (gold) : ``rag-datasets/rag-mini-bioasq``         (relevant_passage_ids)
No large data is committed; everything streams from the HF cache.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

from ..config import DataConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _first_col(ds, candidates: List[str]) -> Optional[str]:
    cols = set(ds.column_names)
    for c in candidates:
        if c in cols:
            return c
    return None


def load_squad(cfg: DataConfig):
    """Load SQuAD v2 (train/validation) for the extractive reader."""
    from datasets import load_dataset

    logger.info("Loading reader dataset: %s", cfg.reader_dataset)
    return load_dataset(cfg.reader_dataset)


def load_retriever_pairs(cfg: DataConfig, limit: Optional[int] = None):
    """Load (query, positive passage) pairs for retriever fine-tuning."""
    from datasets import load_dataset

    logger.info("Loading retriever pairs: %s (%s)", cfg.retriever_pairs_dataset, cfg.retriever_pairs_config)
    try:
        ds = load_dataset(cfg.retriever_pairs_dataset, cfg.retriever_pairs_config, split="train")
    except Exception:
        ds = load_dataset(cfg.retriever_pairs_dataset, split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    q_col = _first_col(ds, ["query", "question", "anchor", "sentence1"])
    p_col = _first_col(ds, ["answer", "passage", "positive", "sentence2", "context"])
    if not q_col or not p_col:
        raise KeyError(f"Cannot find query/passage columns in {ds.column_names}")
    return ds, q_col, p_col


def load_demo_kb(cfg: DataConfig, limit_corpus: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """Return (corpus_docs, qa_pairs) for the demo knowledge base."""
    from datasets import load_dataset

    logger.info("Loading demo KB corpus: %s/%s", cfg.demo_kb_dataset, cfg.demo_kb_corpus_config)
    corpus = load_dataset(cfg.demo_kb_dataset, cfg.demo_kb_corpus_config, split="passages")
    text_col = _first_col(corpus, ["passage", "text", "content", "document"])
    id_col = _first_col(corpus, ["id", "pid", "passage_id"])
    docs: List[Dict] = []
    rng = range(min(limit_corpus, len(corpus))) if limit_corpus else range(len(corpus))
    for i in rng:
        row = corpus[i]
        docs.append({"id": str(row.get(id_col, i)) if id_col else str(i),
                     "title": "", "text": row[text_col]})

    qa = load_dataset(cfg.demo_kb_dataset, cfg.demo_kb_qa_config, split="test")
    q_col = _first_col(qa, ["question", "query"])
    a_col = _first_col(qa, ["answer", "answers"])
    qa_pairs = [{"question": qa[i][q_col], "answer": qa[i][a_col]} for i in range(len(qa))]
    logger.info("Demo KB: %d passages, %d QA pairs", len(docs), len(qa_pairs))
    return docs, qa_pairs


def load_recall_dataset(cfg: DataConfig) -> Tuple[List[Dict], List[Dict]]:
    """Return (corpus_docs, qa_with_gold_ids) for retrieval-recall evaluation."""
    from datasets import load_dataset

    corpus = load_dataset(cfg.recall_dataset, "text-corpus", split="passages")
    text_col = _first_col(corpus, ["passage", "text", "content"])
    id_col = _first_col(corpus, ["id", "pid", "passage_id"])
    docs = [{"id": str(corpus[i].get(id_col, i)) if id_col else str(i), "title": "", "text": corpus[i][text_col]}
            for i in range(len(corpus))]

    qa = load_dataset(cfg.recall_dataset, "question-answer-passages", split="test")
    q_col = _first_col(qa, ["question", "query"])
    a_col = _first_col(qa, ["answer", "answers"])
    rel_col = _first_col(qa, ["relevant_passage_ids", "relevant_passages", "passage_ids"])
    qa_pairs = []
    for i in range(len(qa)):
        rel = qa[i].get(rel_col) if rel_col else None
        if isinstance(rel, str):
            try:
                rel = [str(x) for x in ast.literal_eval(rel)]
            except Exception:
                rel = []
        elif isinstance(rel, list):
            rel = [str(x) for x in rel]
        else:
            rel = []
        qa_pairs.append({"question": qa[i][q_col], "answer": qa[i].get(a_col), "relevant_ids": rel})
    return docs, qa_pairs


__all__ = ["load_squad", "load_retriever_pairs", "load_demo_kb", "load_recall_dataset"]
