# Problem Definition Document

**Project #3 — Knowledge Base Question-Answering System (`kbqa`)**
**Author:** Le Dinh Minh Quan (23127460) · NLP-in-Industry Final Assignment · 2026-06-26
**Scope:** Assignment §2 — Business Context, Stakeholders, Problem Statement, Why NLP/RAG, Success Metrics.

---

## 1. Business Context & Motivation

Modern enterprises are drowning in unstructured text. Product documentation, internal
wikis, runbooks, policy PDFs, onboarding guides, and an ever-growing backlog of support
tickets accumulate far faster than any team can curate. The knowledge needed to answer a
given question almost always *exists* somewhere in the corpus — but finding it is slow,
and the answer is often stale, inconsistent, or scattered across several documents.

This creates three concrete, recurring costs:

1. **Wasted employee time.** Staff spend a substantial share of each day searching across
   tools, re-reading long documents, and pinging colleagues for answers that are already
   written down. Keyword search returns *documents*, not *answers*, pushing the synthesis
   work back onto the human.
2. **Stale and unsourced answers.** When an answer is finally located, it is frequently
   from an outdated revision, and the person consuming it has no easy way to verify where
   it came from. Decisions in support, compliance, and analytics get made on shaky ground.
3. **Hallucination risk of plain LLMs.** The naive fix — "just ask a chatbot" — replaces a
   *retrieval* problem with a *trust* problem. A bare LLM answers from frozen parametric
   memory: it does not know the company's private documents, cannot cite a source, has no
   notion of "this changed last week," and confidently fabricates plausible-sounding
   answers when it does not know. In regulated or customer-facing settings a fluent wrong
   answer is worse than no answer.

The motivating insight is that the *information* is not missing — the **access path** is.
What is needed is a system that reads the organisation's own documents, answers questions
in natural language, **shows its sources**, stays current as documents change *without
retraining*, and — critically — **declines to answer when the corpus does not support
one**, rather than guessing.

```
        Plain keyword search          Plain LLM                  This system (RAG)
        --------------------          ---------                  -----------------
  Q --> ranked DOCUMENTS         Q --> fluent ANSWER         Q --> grounded ANSWER
        (user must read)              (no source, may              + citations
        no synthesis                  hallucinate, stale)         + confidence
        no answer                     no abstention               OR "I don't know"
```

## 2. Target Users & Stakeholders

| Stakeholder | Primary need | What the system gives them |
|---|---|---|
| **Employees** (general knowledge workers) | Fast, trustworthy answers from internal wikis/docs without manual searching | NL answers + cited source spans, lower time-to-answer |
| **Support agents** | Resolve customer tickets quickly using product docs and prior tickets | Grounded answers with citations they can quote, faster handle time |
| **Analysts** | Pull specific facts and multi-hop relationships across many documents | Decompose-and-retrieve multi-hop answers with provenance |
| **Compliance / legal / risk** | Auditable, source-backed answers; zero tolerance for fabricated claims | Faithfulness gate, citation trail, explicit abstention over hallucination |
| **System owners / ML & platform team** | Deployable, CPU-default, zero-paid-API, versioned service | FastAPI/Gradio service, FAISS persistence, model-version pinning |

The common thread across all five groups is **provenance and safety**: every answer must
be traceable to a source passage, and the system must prefer "I don't know" to a guess.

## 3. Problem Statement

> **Given a private knowledge base of documents (PDF / Markdown / HTML / plain text) and a
> natural-language question, return a concise answer that is *grounded* in — and *cited
> back to* — specific passages of that knowledge base, together with a confidence score;
> and when the knowledge base does not contain sufficient supporting evidence, *abstain*
> with an explicit "I don't know" rather than fabricate an answer.**

Formally, the task is open-domain **retrieval-augmented question answering over a private
document corpus**, with two non-negotiable safety properties:

- **Groundedness with citations** — every emitted answer is supported by, and points to,
  retrieved source chunks (`source`, `chunk_id`, span).
- **Safe abstention** — answerability is a first-class output. Unanswerable or
  insufficiently-supported questions yield a refusal, not a hallucination.

The system also handles **multi-hop** questions (answers requiring evidence joined across
several documents) via query decomposition and iterative retrieval. This is implemented as
**RAG-over-documents** — ingest → chunk → index (FAISS) → analyze query → retrieve
(BM25 + BGE dense, fused via RRF) → rerank (cross-encoder) → sufficiency check (CRAG
correction loop) → grounded generate with citations → faithfulness check → answer +
citations + confidence, **or** abstain. The design is **CPU-default and zero-paid-API**.

*(Note: heavyweight semantic-parsing-over-Freebase — ChatKBQA — was deliberately rejected
as the primary approach because it is impractical to deploy over arbitrary enterprise
documents; it is retained only as a pluggable `kg_query` backend for customers who own a
stable knowledge graph.)*

## 4. Why NLP / RAG Is Required

A non-NLP solution (keyword/BM25 search alone) cannot meet the requirements, and a plain
LLM alone is unsafe. RAG is required because it is the only approach that satisfies all
three constraints simultaneously:

1. **Semantics, not keywords.** Users phrase questions differently from how documents are
   written ("vacation policy" vs. "paid time off accrual"). Dense semantic retrieval
   (`BAAI/bge-base-en-v1.5`) matches *meaning*; fusing it with sparse BM25 via Reciprocal
   Rank Fusion recovers exact-entity matches that dense retrieval misses. A cross-encoder
   reranker then maximises top-k precision. Understanding the question and reading the
   passage to extract a span are inherently NLP tasks.
2. **Grounding + provenance.** Retrieval supplies the exact source passages that the reader
   must answer *from*, making citations and an auditable trace possible — something a bare
   LLM cannot provide. A downstream faithfulness/entailment check enforces that the answer
   actually follows from the cited context, enabling principled abstention.
3. **Up-to-date knowledge without retraining.** New or revised documents are *ingested and
   re-embedded*, not learned into model weights. The knowledge base can change daily; the
   models stay fixed. This decoupling of *knowledge* (the index) from *capability* (the
   models) is the core operational advantage of RAG over fine-tuning an LLM on the corpus.

## 5. Success Metrics

Success is measured on two axes — the **business** outcomes that justify the project, and
the **technical** metrics that the engineering team optimises and reports.

### 5.1 Business Metrics

| Business metric | Definition | Target direction |
|---|---|---|
| **Deflection rate** | Share of questions answered directly by the system without escalation to a human | ↑ (more self-service) |
| **Time-to-answer reduction** | Reduction in median time for a user to obtain a trustworthy, sourced answer vs. manual search | ↓ (faster) |
| **Answer trust / citation rate** | Fraction of answers delivered with valid, verifiable citations users can act on | ↑ (toward 100% of answered queries cited) |
| **Abstain-instead-of-hallucinate** | The system says "I don't know" on unsupported questions instead of fabricating | ↑ (safe refusals); near-zero confident-wrong answers |

These map directly onto the motivation in §1: deflection and time-to-answer attack *wasted
employee time*; citation rate attacks *unsourced/stale answers*; and the
abstain-over-hallucinate property attacks the *hallucination risk* that makes compliance
and support stakeholders distrust plain LLMs.

### 5.2 Technical Metrics

| Dimension | Metric | Evaluation source |
|---|---|---|
| **Retrieval** | Recall@{1,5,10}, NDCG@10, MRR@10 | `InformationRetrievalEvaluator`; gold IDs from `rag-datasets/rag-mini-bioasq` (`relevant_passage_ids`) or SQuAD-derived gold contexts |
| **Answer quality** | Exact Match (EM) / F1 | SQuAD v2 scorer (extractive `deepset/roberta-base-squad2`) / normalized EM/F1 (generative `google/flan-t5-base`) |
| **Faithfulness / groundedness** | % of answer content entailed by cited passages | NLI / entailment or embedding-overlap proxy over (answer, cited chunk) |
| **Abstention** | **NoAns-F1**, no-answer precision/recall, abstain rate | SQuAD v2 unanswerables (null-score threshold); abstain rate on unanswerable questions |
| **Latency** | **p50 / p95** per endpoint | `/metrics` histograms — `/ask` extractive target **p50 ~350 ms / p95 ~800 ms** on CPU |

**Decision thresholds (verified):** the agent's sufficiency loop uses `TAU_HIGH = 0.55`
(SUFFICIENT) and `TAU_LOW = 0.15` (INSUFFICIENT), with the CRAG correction loop bounded by
`max_iterations = 3`. The extractive reader abstains via its native null-score threshold;
the multi-hop synthesis applies a final global faithfulness gate before any answer is
emitted.

**Baseline-to-beat.** A floor is established before fine-tuning — BM25 alone vs. zero-shot
`bge-base-en-v1.5` vs. hybrid RRF vs. the fine-tuned retriever (on Recall@k / NDCG@10 /
MRR@10), and BM25 + zero-shot reader (no rerank, no agent loop) as the end-to-end floor.
The full stack (hybrid retrieve → rerank → fine-tuned reader → agent loop) must improve
EM/F1, faithfulness, and citation accuracy while holding CPU latency within the targets
above.

---

*This document defines the problem and success criteria; the architecture, datasets,
models, training recipe, and deployment design are specified in the companion Design Brief
(`docs/DESIGN_BRIEF.md`).*
