# System Architecture

**Project:** Knowledge Base Question-Answering System (`kbqa`) — production, agentic RAG-over-documents.
**Author:** Le Dinh Minh Quan (23127460). **Package:** `kbqa`. **Repo root:** `D:/NLP Industry Projects/03_Knowledge_Base_QA`.

This document describes the system architecture of `kbqa`: the components and their responsibilities, the data-flow for the three operating modes (ingest, ask, training), the model/index versioning and manifest contract, the configuration system, and the mapping from logical components to the `src/kbqa` package modules. All datasets, models, thresholds, and latency figures are taken verbatim from `docs/DESIGN_BRIEF.md`.

---

## 1. Architectural Principles

The design is shaped by four hard constraints, each of which forces a concrete structural decision:

| Principle | Consequence in the architecture |
|---|---|
| **CPU-default, zero paid API** | Every node has a no-LLM heuristic default; GPU and any LLM brain are config-gated upgrades, never required. |
| **Grounded or abstain** | A faithfulness gate sits *after* generation; if the answer is not entailed by retrieved context the system returns "I don't know" rather than fabricate. |
| **Auditable** | `AgentState` threads through every step and is fully serialisable; the API can return the complete reasoning `trace` plus the citations behind each answer. |
| **Pluggable backends** | Retrieval, reranking, and reading are tools behind a stable agent interface; a `kg_query` (ChatKBQA-style NL→SPARQL) backend can be swapped in beside `retrieve` without touching the loop. |

The agent is a **deterministic state machine** (CRAG correction loop + Self-RAG reflection + query rewrite), not a free-form LLM agent. This keeps latency, cost, and behaviour predictable on CPU.

---

## 2. Component Diagram

```
                          ┌──────────────────────────────────────────────────────────┐
                          │                    CLIENTS                                │
                          │   Gradio UI (demo.py)        HTTP clients / batch          │
                          └───────────────┬───────────────────────┬──────────────────┘
                                          │                       │
                                          ▼                       ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  API LAYER  (kbqa.api / app)                                                   │
        │  FastAPI:  /health  /ingest  /search  /ask  /batch  /metrics                   │
        │  in-process model singletons + LRU caches · model_versions echoed in responses │
        └───────┬───────────────────────┬───────────────────────────┬───────────────────┘
                │ /ingest                │ /search · /ask            │ /metrics
                ▼                        ▼                           ▼
   ┌────────────────────┐   ┌───────────────────────────────┐   ┌──────────────────────┐
   │ INGESTION          │   │ AGENT LOOP (kbqa.agent)        │   │ MONITORING           │
   │ (kbqa.data)        │   │ deterministic state machine    │   │ (kbqa.monitoring)    │
   │ preprocessing      │   │                                │   │ Prometheus metrics   │
   │  → chunking        │   │ analyze_query ──► route        │   │ p50/p95, abstain     │
   └─────────┬──────────┘   │      │                         │   │ rate, cache hit      │
             │              │      ▼                         │   └──────────────────────┘
             ▼              │   retrieve ◄───────────────┐   │
   ┌────────────────────┐   │      │  (CRAG loop:         │   │
   │ EMBEDDING          │   │      ▼   rewrite + widen    │   │
   │ (kbqa.models)      │   │   rerank   top_k, ≤3 iters) │   │
   │ bge-base-en-v1.5   │   │      │                      │   │
   │ (MiniLM fallback)  │   │      ▼                      │   │
   └─────────┬──────────┘   │   check_sufficiency ────────┘   │
             │              │      │ SUFFICIENT                │
             ▼              │      ▼                           │
   ┌────────────────────┐   │   generate (reader)             │
   │ VECTOR STORE       │◄──┤      │  extractive / FLAN-T5     │
   │ (kbqa.index)       │   │      ▼                           │
   │ FAISS IndexFlatIP  │   │   check_faithfulness ──► gate    │
   │ + numpy fallback   │   │      │ supported   │ not supported│
   │ + BM25 (kbqa.      │   │      ▼             ▼              │
   │   models.baseline) │   │  answer+cites   abstain "IDK"    │
   └─────────┬──────────┘   └──────────────┬──────────────────┘
             │                             │
             ▼                             ▼
   ┌──────────────────────────────────────────────────────────────────────────────────┐
   │ STORAGE  (filesystem / shared mmap volume)                                          │
   │  kb/{index.faiss, passages.pkl, store_meta.json}  ·  manifest model_version         │
   │  models/<task>/<version>/{weights, model_metadata.json}                             │
   └──────────────────────────────────────────────────────────────────────────────────┘

   TRAINING (offline, H100; kbqa.training)  ──build──►  fine-tuned retriever / reader / FLAN-T5
        consumes datasets (kbqa.data) → writes versioned model dirs → re-index → serve
```

The boxed components above are the eleven logical subsystems. Section 6 maps each to a `src/kbqa` package.

---

## 3. Component Responsibility Table

| Component | Responsibility | Key models / structures | Module |
|---|---|---|---|
| **Ingestion / chunking** | Clean raw text; split documents into overlapping word-windows; emit `Passage` records with `doc_id/chunk_id/title/meta`; dedup by content hash for idempotent re-ingest | `chunk_document`, `chunk_corpus`, SHA-1 chunk ids; default 512-tok target / 64 overlap (brief), `Passage` | `kbqa.data` |
| **Embedding** | Encode queries (with bge query prefix) and passages into normalised 768-d vectors | `BAAI/bge-base-en-v1.5` (MIT); CPU fallback `all-MiniLM-L6-v2` (384-d) | `kbqa.models` |
| **FAISS vector store** | Dense ANN over passage embeddings; cosine via inner product on L2-normalised vectors; incremental `add`; persist/load with manifest assertion | `IndexFlatIP` (<100k) / `IndexHNSWFlat(M=32)` (≥100k); numpy brute-force fallback | `kbqa.index` |
| **BM25 sparse baseline** | Lexical retrieval for exact-term / rare-entity matches; mandatory baseline the dense path must beat; zero-dep fallback | `rank_bm25` BM25Okapi; TF-IDF cosine fallback | `kbqa.models` |
| **Reranker** | Re-score retriever's top-k (20–50) down to top-5; highest-ROI accuracy lever | `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU); `bge-reranker-v2-m3` (GPU swap) | `kbqa.models` |
| **Readers** | Produce grounded answer + span citations from reranked context; native abstention | extractive `deepset/roberta-base-squad2` (null-score abstain); generative `google/flan-t5-base` (grounded + IDK) | `kbqa.models` |
| **Agent loop** | Deterministic state machine: analyze/route, CRAG sufficiency loop (≤3 iters), faithfulness gate; threads `AgentState`, logs `trace` | `AgentState`, `RetrievedPassage`, `ToolTrace`; `TAU_HIGH=0.55`, `TAU_LOW=0.15`, `max_iterations=3` | `kbqa.agent` |
| **API** | Expose `/health /ingest /search /ask /batch /metrics`; hold model singletons + caches; echo `model_versions` | FastAPI; request/response schemas per brief §4.2 | `kbqa.api` (+ `app/`) |
| **UI** | Demo front-end over `/ask`; show answer + confidence + sources | Gradio `demo.py` | `app/` |
| **Monitoring** | Prometheus metrics: request counts, p50/p95 latency, cache-hit rate, `abstain_rate`, `index_n_vectors`, `model_version` | `prometheus-fastapi-instrumentator` | `kbqa.monitoring` |
| **Storage** | Persist FAISS index + passages + store meta; versioned model dirs with metadata; blue/green index dirs | `index.faiss`, `passages.pkl`, `store_meta.json`, `model_metadata.json`, `manifest.json` | `kbqa.index`, `kbqa.models` |
| **Training** | Offline fine-tune retriever (MNRL + hard negatives), extractive reader (SQuAD v2 no-answer), FLAN-T5; write versioned models + metrics | H100 recipe (brief §5); `sentence-transformers==5.6.0` | `kbqa.training` |
| **Evaluation / grading** | Retrieval Recall@k/NDCG@10/MRR@10; reader EM/F1 + NoAns-F1; faithfulness, citation accuracy, abstain rate | `InformationRetrievalEvaluator`, `squad_v2` scorer | `kbqa.grading`, `kbqa.analysis` |

---

## 4. Data Flow

### 4.1 Ingest path (`POST /ingest`)

```
documents[] ──► clean_text ──► chunk_document (512-tok / 64 overlap)
   ──► Passage[] (doc_id, chunk_id, title, hash, meta)
   ──► SHA dedup (skip duplicate chunks)
   ──► embed (bge-base-en-v1.5, passages raw)  ──► L2-normalise
   ──► VectorStore.add(emb, passages)           (FAISS IndexFlatIP)
   ──► VectorStore.save(kb/)                     (index.faiss + passages.pkl + store_meta.json)
   ──► resp { ingested_docs, new_chunks, skipped_duplicate_chunks, index_n_vectors,
              model_version, took_ms }
```

Ingest is **append-only and idempotent**: re-ingesting the same document re-derives identical chunk ids and is skipped. The encoder used here is pinned by `MODEL_VERSION`; the resulting index records that version so a mismatched encoder can never query it (Section 5).

### 4.2 Ask path (`POST /ask`) — the agentic core

```
question
  │
  ▼  analyze_query        → {qtype, rewritten, sub_questions}; route simple|multihop|unanswerable
  │                          (unanswerable ⇒ immediate "I don't know")
  ▼  retrieve (top_k=20)  → FAISS dense (+ optional BM25 ⊕ RRF) → RetrievedPassage[]
  ▼  rerank (top_n=5)     → cross-encoder ms-marco-MiniLM-L-6-v2 → .rerank_score
  ▼  check_sufficiency    → top rerank ≥ TAU_HIGH(0.55) SUFFICIENT
  │                          [TAU_LOW,TAU_HIGH) AMBIGUOUS  ── rewrite + widen top_k, retry ──┐
  │                          < TAU_LOW(0.15) INSUFFICIENT  ── decompose / clarify ───────────┤
  │                                                                                          │
  │   ◄──────────────── CRAG correction loop, bounded by max_iterations=3 ───────────────────┘
  ▼  generate             → extractive roberta-base-squad2 (span + source chunk)
  │                          | optional FLAN-T5 "answer using ONLY context; cite [chunk_id]"
  ▼  check_faithfulness    → supported (entailment ≥ τ) ? emit : abstain "I don't know"
  ▼
  resp { answer, citations[{marker, chunk_id, doc_id, source, quote, offset}],
         confidence, is_answerable, trace{steps, model_version}, timing_ms }
```

Per the brief, for **multi-hop** questions the loop runs per sub-question (fill dependent slots from prior hops → retrieve → rerank → sufficiency → generate → faithfulness), then **synthesises**: join sub-answers, dedup citations, `confidence = min` over hops, and run a final global faithfulness gate. Any required hop failing after `max_iterations` ⇒ abstain (optionally a supported partial answer). Every branch is appended to `AgentState.trace`. Abstention triggers: query routed unanswerable, top rerank below threshold, or extractive null-score wins.

**CPU latency targets** (single replica, from brief §4.3): `/ask` extractive **p50 350 ms / p95 800 ms**; `/ask` FLAN-T5-base **0.9 / 2.0 s**; `/search` (k=20, 50→8) 120 / 300 ms; `/health` <5 ms.

### 4.3 Training path (offline, H100)

```
datasets (kbqa.data: squad_v2, natural-questions, hotpot_qa, rag-mini-*, rag-dataset-12000)
  ├─ Retriever: mine_hard_negatives → CachedMultipleNegativesRankingLoss (bf16, tf32)
  │             → re-mine with fine-tuned model → retrain  → versioned retriever dir
  ├─ Extractive reader: SQuAD v2 (doc_stride=128, no-answer→[CLS], version_2_with_negative)
  │             → EM/F1 + HasAns-F1/NoAns-F1                → versioned reader dir
  └─ FLAN-T5 (optional): grounded + "say I don't know" supervision (bf16 only; fp16 NaNs)
                                                            → versioned generative dir
  each writes model_metadata.json (base_model, config, dataset, metrics, env, git_sha)
  ──► rebuild FAISS index with new retriever ──► bump MODEL_VERSION ──► serve
```

All three stages are resume-safe (`trainer.train(resume_from_checkpoint=True)`, `save_total_limit=3`). Training **closes the train/serve gap** by re-indexing after retriever fine-tuning and training the reader on *retrieved* (not gold) contexts.

---

## 5. Model + Index Versioning and Manifest

The single most important operational invariant: **embeddings produced by different encoders are not cross-compatible**, so an index must never be queried by a different encoder than built it.

- **`MODEL_VERSION`** (env) pins the encoder + reranker + reader + index **together** as one logical bundle. It is echoed in every API response and in `/metrics`.
- **Index manifest.** On build the store writes a manifest (`store_meta.json` today; brief `manifest.json` contract: `{model_version, dim, metric, n_vectors, built_at}`). On load, the service **asserts `manifest.model_version == MODEL_VERSION`** and refuses a cross-version index.
- **Model metadata.** Each trained model dir carries `model_metadata.json` via `kbqa.models.model_registry.save_model_metadata`: `created_at, task, base_model, version, config, dataset, metrics, environment` (Python/torch/transformers/datasets/sentence-transformers/faiss versions + `git_sha`). `resolve_latest` + `has_model` locate the newest usable checkpoint.
- **Blue/green index dirs.** `/kb/v1`, `/kb/v2` allow zero-downtime swaps: build the new index offline, publish, hot-swap stateless replicas. HNSW supports adds but not deletes → deletes are tombstoned and compacted by an offline rebuild.

```
MODEL_VERSION = "bge-base+msmarco-mini6+roberta-squad2@2026-06-26"
                 └ encoder ──┘ └ reranker ─┘ └── reader ──┘  └ build date ┘
   ▲ asserted equal to ▼
manifest.json (kb/) : { model_version, dim:768, metric:"ip", n_vectors, built_at }
```

---

## 6. Configuration System

Configuration is layered, with later layers overriding earlier ones, so the same code runs as a CPU demo or a GPU service by changing config only:

1. **Defaults in code** — thresholds and sizes baked as constants: `TAU_HIGH=0.55`, `TAU_LOW=0.15`, `max_iterations=3`, `top_k=20`, `rerank_top_n=5`, chunk size/overlap.
2. **`configs/`** — declarative YAML/TOML selecting model IDs, retriever/reranker/reader choice, hybrid (BM25⊕RRF) on/off, LLM-brain plug-in toggles for the three upgradeable nodes (decompose, sufficiency, faithfulness).
3. **Environment** — `MODEL_VERSION`, `OMP_NUM_THREADS`, device flags, index dir (`/kb/vN`); these drive deployment (Docker/HF Space) and the manifest assertion.
4. **Request-level** — per-call overrides on `/search` and `/ask` (`top_k`, `rerank_top_n`, `reader`, `max_new_tokens`, `require_citations`, `return_trace`).

Every node has a **no-LLM heuristic default**; the optional local LLM brain is purely additive at three nodes:

| Node | No-LLM default (CPU) | Optional LLM upgrade |
|---|---|---|
| decompose | rule-based + thesaurus | few-shot decomposition |
| sufficiency | threshold + coverage | Self-RAG ISREL |
| generate | extractive span / FLAN-T5 | instruct LLM, cite chunk_ids |
| faithfulness | NLI / overlap | Self-RAG ISSUP |

---

## 7. Mapping to `src/kbqa` Package Modules

The package lives under `src/kbqa` (src-layout, `__version__ = "1.0.0"`). Eleven subpackages plus a shared `logging_utils.py`:

| Package | Role | Status (implemented files) |
|---|---|---|
| `kbqa.data` | Ingestion: `preprocessing.clean_text`, `chunking.{chunk_text,chunk_document,chunk_corpus}`, `samples` | implemented |
| `kbqa.index` | `vector_store.{VectorStore,Passage}` — FAISS `IndexFlatIP` + numpy fallback + save/load | implemented |
| `kbqa.models` | `baseline_bm25.BM25Retriever`, `model_registry` (metadata/versioning); embedding/reranker/reader wrappers | partial (registry + BM25 implemented) |
| `kbqa.agent` | `state.{AgentState,RetrievedPassage,ToolTrace,AnswerStatus}`; tools + state-machine loop | partial (state implemented) |
| `kbqa.api` | FastAPI app: `/health /ingest /search /ask /batch /metrics`, singletons, caches | scaffolded |
| `kbqa.monitoring` | Prometheus metrics, latency histograms, abstain-rate | scaffolded |
| `kbqa.training` | Retriever / extractive-reader / FLAN-T5 fine-tune entrypoints (H100 recipe) | scaffolded |
| `kbqa.grading` | Metric computation: Recall@k/NDCG/MRR, EM/F1/NoAns-F1, citation accuracy | scaffolded |
| `kbqa.analysis` | Error analysis, baseline comparisons, ablations | scaffolded |
| `kbqa.autoreport` | Report-artifact generation (tables/figures feeding the PDF) | scaffolded |
| `kbqa.automation` | Pipeline orchestration / make targets glue (e.g. `data-demo`, index build) | scaffolded |
| `kbqa.logging_utils` | `get_logger`, `utc_now_iso` — shared logging + timestamps | implemented |

Supporting top-level dirs: `app/` (FastAPI `app.py` + Gradio `demo.py`), `configs/`, `deploy/` (Dockerfile, HF Space — port 7860), `scripts/` (`download_data.py`, index build), `notebooks/` (H100 training), `tests/`, `models/` and `data/` (gitignored artifacts; both carry a `README.md`).

> **Implementation note.** The dataclasses already in place encode the architecture's contracts: `Passage` (the storage record), `RetrievedPassage`/`ToolTrace`/`AgentState` (the auditable loop), and `VectorStore` (the manifest-bearing index). The remaining packages are scaffolded stubs that the components above will fill, each behind the stable tool/endpoint boundaries described here.
