"""Built-in synthetic knowledge base + QA pairs for tests, demo and CI.

A tiny, self-contained KB (general factual snippets) so the full RAG pipeline can
be exercised offline — no dataset download required. Answers are grounded in the
documents below, which lets tests assert retrieval + answering behaviour.
"""

from __future__ import annotations

from typing import Dict, List

SAMPLE_DOCS: List[Dict] = [
    {"id": "doc_python", "title": "Python (programming language)",
     "text": "Python is a high-level, general-purpose programming language created by Guido van Rossum "
             "and first released in 1991. Python emphasizes code readability and uses significant "
             "indentation. It supports multiple programming paradigms, including procedural, "
             "object-oriented and functional programming."},
    {"id": "doc_transformer", "title": "Transformer (machine learning)",
     "text": "The Transformer is a deep learning architecture introduced in the 2017 paper "
             "'Attention Is All You Need' by Vaswani and colleagues. It relies on a self-attention "
             "mechanism and dispenses with recurrence and convolutions entirely. Transformers are the "
             "foundation of modern large language models such as BERT and GPT."},
    {"id": "doc_faiss", "title": "FAISS",
     "text": "FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta for "
             "efficient similarity search and clustering of dense vectors. It supports exact and "
             "approximate nearest-neighbour search and can scale to billions of vectors, making it "
             "popular for retrieval-augmented generation systems."},
    {"id": "doc_rag", "title": "Retrieval-Augmented Generation",
     "text": "Retrieval-Augmented Generation (RAG) is a technique that combines a retriever with a "
             "generative model. The retriever fetches relevant documents from a knowledge base, and the "
             "generator produces an answer grounded in those documents. RAG reduces hallucination and "
             "lets a model use up-to-date external knowledge without retraining."},
    {"id": "doc_bm25", "title": "BM25",
     "text": "BM25 is a ranking function used by search engines to estimate the relevance of documents "
             "to a query. It is a bag-of-words retrieval function based on term frequency and inverse "
             "document frequency, and remains a strong sparse-retrieval baseline."},
    {"id": "doc_squad", "title": "SQuAD",
     "text": "The Stanford Question Answering Dataset (SQuAD) is a reading-comprehension dataset of "
             "questions posed on Wikipedia articles, where the answer to every question is a span of "
             "text from the corresponding passage. SQuAD 2.0 adds unanswerable questions."},
]

# (question, expected-answer-substring, supporting doc_id) — grounded in SAMPLE_DOCS.
SAMPLE_QA: List[Dict] = [
    {"question": "Who created the Python programming language?", "answer": "Guido van Rossum", "doc_id": "doc_python"},
    {"question": "In what year was the Transformer architecture introduced?", "answer": "2017", "doc_id": "doc_transformer"},
    {"question": "What does FAISS stand for?", "answer": "Facebook AI Similarity Search", "doc_id": "doc_faiss"},
    {"question": "What problem does Retrieval-Augmented Generation reduce?", "answer": "hallucination", "doc_id": "doc_rag"},
    {"question": "What does SQuAD 2.0 add compared to SQuAD?", "answer": "unanswerable questions", "doc_id": "doc_squad"},
    {"question": "What is the capital of France?", "answer": None, "doc_id": None},  # not answerable from the KB
]

__all__ = ["SAMPLE_DOCS", "SAMPLE_QA"]
