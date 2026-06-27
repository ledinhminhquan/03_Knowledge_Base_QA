# Project Management & Teamwork

**Project #3 — Knowledge Base Question-Answering System (`kbqa`)**
Author: Le Dinh Minh Quan (23127460). Solo execution; team roles simulated to demonstrate how the work would partition in a real engineering organization. This document satisfies Assignment §10: a phased timeline, a role-mapped task breakdown, a risk register, and a reflection on scaling the system inside a real team.

The system under management is a production RAG-over-documents pipeline: **ingest/chunk/index (FAISS) → analyze query → retrieve (BM25 + bge dense, RRF) → rerank (cross-encoder) → sufficiency check (CRAG loop) → grounded generate with citations → faithfulness check → answer + citations + confidence, or abstain "I don't know."** It is CPU-default and requires zero paid API. ChatKBQA-style semantic parsing over Freebase is kept as a documented, pluggable `kg_query` backend rather than the default path.

---

## 1. Phased Timeline (§10)

The plan spans **7 weeks (≈8 with the report buffer)**. Each phase ends in a concrete, verifiable artifact and a go/no-go checkpoint, so the project degrades gracefully — every week ships something demonstrable even if a later phase is cut.

| Wk | Phase | Scope / objective | Primary modules touched | Exit artifact (gate) |
|----|-------|-------------------|--------------------------|----------------------|
| 1 | **Scope & data** | Lock the RAG-over-documents decision; verify every dataset ID/config/split/license live on the Hub; write `download_data.py`; build the SQuAD-derived zero-download fallback (B4/C4). | `scripts/download_data.py`, `data/.gitignore`, `docs/DESIGN_BRIEF.md` | All 10 datasets verified; `make data-demo` materializes `rag-mini-wikipedia` (3,200 passages / 918 QA). |
| 2 | **Chunking & index** | Recursive splitter (512 tok / 64 overlap), per-chunk metadata + SHA-256 dedup; build FAISS (`IndexFlatIP` <100k, `IndexHNSWFlat M=32` ≥100k) with `manifest.json`. | `kbqa.ingest`, `kbqa.index` | Idempotent re-ingest; demo FAISS index + `meta.parquet` + `manifest.json`. |
| 3 | **Retriever** | Fine-tune `BAAI/bge-base-en-v1.5` (MNRL + hard negatives) on `sentence-transformers/natural-questions`; BM25 (`rank_bm25`) + dense + RRF hybrid; MiniLM CPU fallback. | `kbqa.retriever`, `train/retriever.py` | Recall@{1,5,10}/NDCG@10/MRR@10 on `rag-mini-bioasq` gold IDs; fine-tuned beats BM25 floor. |
| 4 | **Reranker & reader** | Wire `cross-encoder/ms-marco-MiniLM-L-6-v2` (top-50→top-5); fine-tune `deepset/roberta-base-squad2` (null-score abstain, `doc_stride=128`); optional FLAN-T5-base grounded reader. | `kbqa.rerank`, `kbqa.reader`, `train/reader_*.py` | EM/F1 + **NoAns-F1** reported separately; reranked + fine-tuned beats zero-shot. |
| 5 | **Agent loop** | Deterministic state machine: 3 decision points (analyze/route, sufficiency CRAG loop `max_iterations=3`, faithfulness gate→abstain); thresholds `TAU_HIGH=0.55`, `TAU_LOW=0.15`; query rewrite/decompose. | `kbqa.agent.{state,tools,loop}` | Multi-hop worked example passes end-to-end; abstention variant never fabricates. |
| 6 | **API & UI** | FastAPI `/health /ingest /search /ask /batch /metrics`; FAISS persistence with `manifest.model_version` assertion; Gradio demo; Docker + HF Space (port 7860); model_versions echoed. | `app.py`, `demo.py`, `Dockerfile` | `/ask` extractive p50/p95 ≈ 350/800 ms CPU; `/metrics` Prometheus live. |
| 7 | **Eval harness** | Full metric suite (retrieval, EM/F1, faithfulness/groundedness, citation accuracy, abstain rate, latency); baseline-vs-full ablation table; calibrate confidence + null-score threshold. | `kbqa.eval`, `scripts/run_eval.py` | Reproducible results table; full stack beats BM25 + zero-shot floor. |
| 8 | **Docs & report** | Consolidate design brief, architecture diagram, results, risks into the 10–15 page PDF; finalize READMEs and reproducibility instructions. | `docs/*`, `README.md` | Submission-ready PDF + tagged repo. |

```
Wk1     Wk2      Wk3        Wk4          Wk5       Wk6      Wk7     Wk8
Scope──►Index──►Retriever─►Rerank+Read─►Agent────►API/UI──►Eval──►Report
 │       │        │            │           │         │       │       │
 data   FAISS   Recall@k    EM/NoAns-F1  CRAG loop  /ask   ablation PDF
 gate   built   beats BM25  null-abstain abstain    @7860  table
```

Critical path runs through retriever → reader → agent → API; the eval harness (Wk7) is partly parallelizable from Wk4 onward because the metric scaffolding only needs a stub pipeline.

---

## 2. Task Breakdown by Simulated Role (§10)

Five roles are simulated. Each owns a vertical slice of the stack and the modules below map one-to-one to the codebase. In a real team these are five people; here one author rotates through the hats, but the separation keeps interfaces clean and ownership explicit.

| Role | Owns (modules) | Key deliverables | Phases led |
|------|----------------|------------------|------------|
| **Data / IR Engineer** | `scripts/download_data.py`, `kbqa.ingest`, `kbqa.index`, `kbqa.retriever` | License-verified data pipeline; chunking + SHA-256 dedup; FAISS build/persist; BM25 + bge + RRF hybrid; hard-negative mining. | 1, 2, 3 |
| **ML Engineer** | `train/retriever.py`, `train/reader_extractive.py`, `train/reader_flan_t5.py`, `kbqa.rerank`, `kbqa.reader` | MNRL retriever fine-tune (resume-safe, H100); `roberta-base-squad2` null-score reader; cross-encoder reranker; optional FLAN-T5 (bf16 only). | 3, 4 |
| **Backend Engineer** | `app.py`, `kbqa.agent.*`, `demo.py` | FastAPI endpoints + schemas; agent state machine + 3 decision points; Gradio UI; abstention contract (`is_answerable:false`). | 5, 6 |
| **MLOps Engineer** | `Dockerfile`, `kb/{kb.index,meta.parquet,manifest.json}`, `/metrics`, HF Space config | Containerization; FAISS persistence + `model_version` assertion; blue/green index dirs; ONNX int8; Prometheus; latency budgets. | 6 |
| **Project Manager** | `docs/*`, eval gating, risk register | Phase gates, timeline, baseline-must-beat policy, risk tracking, final PDF report; cross-role interface contracts. | 1, 7, 8 |

**Module-to-decision-point mapping (agentic core, Backend-owned):**
`analyze_query` (route simple/multi-hop/unanswerable) → `check_sufficiency` (CRAG loop, `max_iterations=3`) → `check_faithfulness` (entailment gate → abstain). The `kg_query` ChatKBQA backend is a pluggable tool behind the same `AgentState` interface, owned jointly by ML + Backend.

---

## 3. Risk Register (§10)

Likelihood (L) / Impact (I) on a 1–3 scale. Mitigations are the verified fallbacks from the design brief — nothing speculative.

| # | Risk | L | I | Mitigation / verified fallback | Owner |
|---|------|---|---|--------------------------------|-------|
| R1 | **Dataset license / availability** — `trivia_qa` license unknown; MS MARCO family is research-terms. | 2 | 3 | Drop to `neural-bridge/rag-dataset-12000` (Apache-2.0) and `Tevatron/msmarco-passage` (Apache-2.0); SQuAD-derived B4/C4 pairs+KB need zero extra download. Legal sign-off before any commercial use of flagged sets. | Data/IR + PM |
| R2 | **Hallucination** — reader invents facts not in context. | 2 | 3 | Grounded reader + `check_faithfulness` entailment gate → abstain "I don't know"; extractive null-score abstention; final global faithfulness gate on synthesized multi-hop answers. Never answer from parametric memory. | Backend + ML |
| R3 | **CPU latency** breaches `/ask` ≈350–800 ms target. | 2 | 2 | ONNX int8 (2–4×) on bi-encoder + cross-encoder; HNSW ANN; rerank only top-50; MiniLM CPU-fallback encoder; LRU query cache; `max_new_tokens=256`. Heavy models (`bge-reranker-v2-m3`, `deberta-v3-large`, `flan-t5-large`) behind GPU config flags. | MLOps |
| R4 | **Cross-version index reuse** — embeddings not cross-compatible after encoder swap. | 1 | 3 | `manifest.model_version` assertion on load (refuse mismatch); blue/green index dirs (`/kb/v1`, `/kb/v2`); re-index on any encoder change. | MLOps |
| R5 | **Multi-hop precision loss vs SPARQL.** | 2 | 2 | Agent decompose + iterative retrieval + sufficiency loop; offer pluggable `kg_query` (ChatKBQA-style) backend for customers with a real KG. | Backend + ML |
| R6 | **Unverified NLI faithfulness model** (`cross-encoder/nli-deberta-v3-small`). | 2 | 2 | Not verified on Hub — do not assert it exists. Fallback to verified `bge-base-en-v1.5` embedding-overlap or `ms-marco-MiniLM-L-6-v2` rerank-score groundedness proxy. | ML |
| R7 | **Tokenizer / training breakage** — DeBERTa-v3 SentencePiece fast tokenizer brittle; FLAN-T5 NaNs under fp16. | 1 | 2 | Pin `transformers`/`tokenizers`; default to `roberta-base-squad2`; train FLAN-T5 in **bf16 only, never fp16**. | ML |
| R8 | **Long documents** exceed encoder context. | 2 | 1 | 512-token chunks / 64 overlap / recursive splitter; reader `doc_stride=128` sliding window; `gte-base-en-v1.5` (8192 ctx) only if truly needed, accepting `trust_remote_code` cost. | Data/IR |
| R9 | **Solo-author single point of failure / scope creep.** | 2 | 2 | Phase gates with shippable artifacts each week; baseline-must-beat policy prevents over-engineering; report buffer in Wk8. | PM |

---

## 4. Reflection: Scaling This in a Real Team

The single-author repo is deliberately structured so it can absorb a team without rewrites. Five concerns dominate the transition from "final assignment" to "service with a pager."

**CI/CD.** The hand-run `make` targets become pipeline stages. A pull request should trigger: lint + unit tests on `kbqa.*`, a fast smoke test of `/health`, `/search`, and `/ask` against the small `rag-mini-wikipedia` index, and a contract test asserting the abstention path (`is_answerable:false` when top rerank < τ or extractive null-score wins). Model and index artifacts ship through a separate, slower pipeline gated on the eval harness — code and weights move on different cadences because retraining is expensive and risky.

**Eval harness as a merge gate.** The Wk7 harness is the team's safety net: no retriever, reader, or prompt change merges unless Recall@k/NDCG@10/MRR@10, EM/F1, **NoAns-F1**, faithfulness, citation accuracy, and abstain rate hold or improve against the committed baseline (the BM25 + zero-shot floor). This converts the solo "must-beat-baseline" policy into an automated regression gate, so that a well-meaning prompt tweak can't silently raise the hallucination rate.

**Index operations.** A real corpus changes daily. The append-only `add_with_ids` + tombstone + periodic `rebuild` design already separates online ingest from offline compaction. At team scale this becomes a dedicated index-ops rotation: one ingest worker funnels writes, publishes a new versioned index, and replicas hot-swap via blue/green dirs with the `manifest.model_version` assertion guarding every load. Re-embedding on encoder upgrades is a planned migration with a parallel index, not an in-place mutation.

**On-call.** `/metrics` (p50/p95 per endpoint, cache hit rate, `index_n_vectors`, **abstain_rate**, `model_version`) is the on-call dashboard. The two alerts that matter most are latency p95 breaching the §4.3 budget and a sudden swing in abstain rate — a spike means retrieval regressed or the corpus drifted; a collapse toward zero may mean the faithfulness gate broke and hallucinations are leaking. Runbooks map each alert to a config lever (rerank depth, ONNX quantization, index rebuild, threshold rollback).

**Data governance.** The license discipline from Wk1 becomes policy: every dataset carries its SPDX tag, flagged sets (`trivia_qa`, MS MARCO family) require legal sign-off before commercial training, and CC-BY/CC-BY-SA sources demand attribution surfaced in product docs. Ingested customer documents need provenance metadata (`doc_id`, `source`, `ingested_at`, content hash), deletion support via tombstones for the right-to-be-forgotten, and PII handling at chunk time. Citations are not just a quality feature — they are the audit trail that makes the system defensible.

In short, the architecture's seams (versioned indexes, model_version pinning, an entailment-gated abstention contract, and a metric-driven eval harness) are exactly the seams a team needs to parallelize safely. The simulated roles in §2 are the org chart this repo is already shaped for.
