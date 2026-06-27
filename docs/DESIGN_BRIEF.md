I'll synthesize the four research briefs into one authoritative design brief. Let me write it directly as my output.

# DESIGN BRIEF — Project #3: RAG-based Knowledge Base QA System (`kbqa`)

**Status of claims:** Every dataset and model ID below is marked VERIFIED only where the four research briefs confirmed it live against the HF Hub (2026-06-26). Anything not verified is explicitly flagged with a verified fallback. Target: runnable Python repo (package `kbqa`) + H100 Colab training notebook + full docs. CPU-default inference, zero paid API.

---

## 0. System Overview

**What it is.** A production RAG-over-documents Knowledge Base QA system. It ingests raw documents (PDF/MD/HTML/text), indexes them in a FAISS vector store, and answers natural-language questions with **grounded, cited** answers — or abstains with "I don't know" when the corpus lacks support. It runs **on CPU by default**; GPU is a speed/accuracy knob, not a requirement. No paid API is ever required.

**Why RAG-over-documents and not semantic parsing (ChatKBQA).** The reference system ChatKBQA (Luo et al., ACL 2024 Findings) is a generate-then-retrieve semantic parser over Freebase: a fine-tuned LLaMA-2 emits an S-expression → logical form, an unsupervised retriever (SimCSE/BM25) grounds mentions to Freebase IDs, and the result executes as SPARQL (WebQSP F1 79.8 / Hits@1 83.2; CWQ F1 77.8 / Hits@1 82.7). It is accurate and auditable (exact SPARQL joins) but **operationally impractical for a general production assistant**: it needs a frozen ~50 GB Freebase dump in a Virtuoso triplestore (100 GB+ RAM), GPU for the parser, and **per-schema fine-tuning** for every new domain. Enterprise knowledge lives in PDFs/wikis/tickets, not curated RDF. We therefore build RAG-over-documents (drop in docs → re-embed, no labels, CPU-default, citations to source spans, safe abstention) and **keep semantic parsing as a documented, pluggable backend** — a `kg_query` tool sits beside `retrieve` behind the same agent interface for customers who do own a stable knowledge graph.

**Pipeline.**

```
ingest/chunk/index → query analysis/decompose → retrieve (FAISS, +BM25/RRF)
  → rerank (cross-encoder) → sufficiency check → grounded generate w/ citations
  → faithfulness check → answer + citations + confidence  |  abstain "I don't know"
```

**Agent.** A deterministic state machine (CRAG-style correction loop + Self-RAG reflection + query rewrite/decompose) drives the pipeline. Every decision node has a **no-LLM heuristic default**; an **optional local LLM brain** (FLAN-T5 / Qwen) upgrades three nodes (decompose, sufficiency, faithfulness). The agent never fabricates from parametric memory — the faithfulness gate requires entailment from retrieved context.

---

## 1. DATA Stack

All IDs below are **VERIFIED live** (HF dataset-viewer, 2026-06-26): IDs, configs, splits, row counts, schemas, licenses confirmed. No large data is committed to the repo — `scripts/download_data.py` pulls on demand via `datasets.load_dataset`; only small derived artifacts (the demo KB index) may be cached.

### 1.1 Reader QA (question, context, answer)

| Role | HF ID | Config / Split (rows) | License | Status |
|---|---|---|---|---|
| **PRIMARY** extractive + abstain | `rajpurkar/squad_v2` | `squad_v2` / train 130,319 · val 11,873 | CC-BY-SA-4.0 | ✅ VERIFIED |
| Multi-hop / agentic | `hotpotqa/hotpot_qa` | `distractor` / train 90,447 · val 7,405 | CC-BY-SA-4.0 | ✅ VERIFIED |
| Generative (optional) | `mandarjoshi/trivia_qa` | `rc.nocontext` / train 138,384 | **unknown** ⚠️ | ✅ VERIFIED — license flagged |

**Decision:** SQuAD v2 is the reader backbone. Its ~50K **unanswerable** questions are the single most important asset for production-grade abstention — the reader learns to point at `[CLS]`/null instead of hallucinating. Add HotpotQA-distractor for multi-hop/agentic evaluation. `trivia_qa` license is `unknown` on HF (research-friendly but not SPDX) — **do not use for commercial training without legal sign-off**; it is optional only.

**SQuAD v2 schema:** `id`(str), `title`(str), `context`(str), `question`(str), `answers`={`text`:list[str], `answer_start`:list[int32]} — empty lists ⇒ unanswerable.

### 1.2 Retriever pairs (question, positive passage)

| Role | HF ID | Config / Split (rows) | License | Status |
|---|---|---|---|---|
| **PRIMARY** fast start | `sentence-transformers/natural-questions` | `pair` / train 100,231 | CC-BY-SA-3.0 | ✅ VERIFIED |
| Scale (triplets, IDs) | `sentence-transformers/msmarco` | `triplets` / 397.2M | MS MARCO terms ⚠️ | ✅ VERIFIED — license flagged |
| Text inlined (hard negs) | `Tevatron/msmarco-passage` | `default` / train 400,782 · val 6,980 | Apache-2.0 | ✅ VERIFIED |

**Decision:** First bi-encoder fine-tune on `sentence-transformers/natural-questions` `pair` (clean `query`/`answer` pairs, no joins, MNRL-ready). Scale to `sentence-transformers/msmarco` `triplets` for stronger general retrieval (IDs — join against its `corpus` 8.84M + `queries` 808,731 configs to materialize text). `Tevatron/msmarco-passage` is the Apache-2.0, text-inlined alternative with pre-mined hard negatives. **MS MARCO family is research-terms / unknown SPDX — flag for legal before commercial use.**

**Zero-download fallback (B4):** derive retriever positives directly from SQuAD v2 — each answerable `(question, context)` is a positive pair, de-dup contexts for the passage pool. Keeps retriever/reader on the same distribution.

### 1.3 Demo / eval RAG corpus + QA (self-contained KB)

| Role | HF ID | Configs (rows) | License | Status |
|---|---|---|---|---|
| **PRIMARY** demo KB | `rag-datasets/rag-mini-wikipedia` | `text-corpus`/passages 3,200 · `question-answer`/test 918 | CC-BY-3.0 | ✅ VERIFIED |
| + retrieval-recall (gold IDs) | `rag-datasets/rag-mini-bioasq` | `text-corpus`/passages 40,200 · `question-answer-passages`/test 4,719 | CC-BY-2.5 | ✅ VERIFIED |
| Generative-RAG fine-tune | `neural-bridge/rag-dataset-12000` | `default` train 9,600 / test 2,400 | **Apache-2.0** ✅ | ✅ VERIFIED |
| IR benchmark (optional) | `BeIR/scifact` | corpus 5,183 · queries 1,109 (+`BeIR/scifact-qrels`) | CC-BY-SA-4.0 | ✅ VERIFIED |

**Decision:** `rag-mini-wikipedia` is the default reproducible KB (index `text-corpus`/`passages`, eval on `question-answer`/`test` — note QA rows carry **no gold passage IDs**, so it measures end-to-end answer quality only). For **retrieval Recall@k / MRR**, use `rag-mini-bioasq` — its `question-answer-passages` rows include `relevant_passage_ids` (string like `"[9797, 11906]"`, parse with `ast.literal_eval`). `neural-bridge/rag-dataset-12000` is the cleanest-license (Apache-2.0) generative-RAG fine-tune set. `BeIR/scifact` (+ separate qrels repo) for rigorous IR NDCG/Recall.

**Build-from-SQuAD fallback (C4):** dedup SQuAD v2 `context` into a ~1.2K-passage KB; each QA item has a known gold context ⇒ exact retrieval-recall measurement, zero extra download.

### 1.4 License summary (production)

- **Cleanest commercial:** `neural-bridge/rag-dataset-12000` (Apache-2.0), `Tevatron/msmarco-passage` (Apache-2.0 repo tag — but derived from MS MARCO data; verify upstream).
- **CC-BY-SA / CC-BY (attribution + share-alike):** SQuAD v2, HotpotQA, the two NQ-derived sets, rag-mini-wikipedia, rag-mini-bioasq, scifact.
- **⚠️ Flag for legal:** `trivia_qa` (unknown), MS MARCO family (`sentence-transformers/msmarco`, `microsoft/ms_marco`) — research terms.

### 1.5 Download scripts

`scripts/download_data.py` exposes `--reader`, `--retriever`, `--demo-kb` flags, each calling `load_dataset(...)` with the exact verified config/split, writing only to the HF cache. The repo commits **no** large data; `data/.gitignore` blocks `*.parquet`/`*.arrow`. A `make data-demo` target materializes only the small `rag-mini-wikipedia` corpus + builds the demo FAISS index.

---

## 2. MODEL Stack

All 13 IDs **VERIFIED live** (HF Hub: existence, param count, license, task tag, 2026-06-26). Hub canonicalizes `cross-encoder/ms-marco-MiniLM-L-6-v2` → display `ms-marco-MiniLM-L6-v2`; **both strings resolve to the same valid repo**.

### 2.1 Retriever — dense bi-encoder

| HF ID | Params | License | Dim | Status |
|---|---|---|---|---|
| **PRIMARY** `BAAI/bge-base-en-v1.5` | 109.5M | **MIT** | 768 | ✅ VERIFIED |
| Alt `intfloat/e5-base-v2` | ~109M | MIT | 768 | ✅ VERIFIED |
| **CPU FALLBACK** `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | Apache-2.0 | 384 | ✅ VERIFIED |
| (long-ctx, avoid by default) `Alibaba-NLP/gte-base-en-v1.5` | 136.8M | Apache-2.0 | 768 / 8192 ctx | ✅ VERIFIED |

**Decision:** `BAAI/bge-base-en-v1.5` primary (MIT = most permissive, strong asymmetric QA retrieval). `all-MiniLM-L6-v2` is the dense CPU fallback (¼ index size, ~4–5× faster CPU encode). **Re-index when switching — embeddings are not cross-compatible.** Avoid `gte-base-en-v1.5` unless 8192-token context is needed (`trust_remote_code=True` = supply-chain/maintainability cost).

**Prefix conventions (getting these wrong silently degrades recall):**
- **bge-base-en-v1.5:** prepend `"Represent this sentence for searching relevant passages: "` to **queries only**; passages embedded raw.
- **e5-base-v2:** prepend `"query: "` to questions and `"passage: "` to documents — mandatory.
- **all-MiniLM / all-mpnet:** symmetric, **no prefixes** — do NOT add e5/bge prefixes.
- All: L2-normalize embeddings, use cosine/dot (equivalent after normalization).

### 2.2 Reranker — cross-encoder

| HF ID | Params | License | Status |
|---|---|---|---|
| **CPU DEFAULT** `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22.7M | Apache-2.0 | ✅ VERIFIED |
| GPU/accuracy-max `BAAI/bge-reranker-v2-m3` | 567.8M | Apache-2.0 | ✅ VERIFIED |

**Decision:** `ms-marco-MiniLM-L-6-v2` reranks top-50→top-5 in single-digit ms/pair on CPU (the CPU-default constraint). `bge-reranker-v2-m3` is a config-swappable GPU upgrade (~25× CPU cost — GPU/offline only). **Reranking is the highest-ROI accuracy lever in RAG — always rerank the bi-encoder's top-k (20–50) down to 3–5.**

### 2.3 Extractive reader (span QA, abstains)

| HF ID | Params | License | Status |
|---|---|---|---|
| **TRAINABLE PRIMARY** `deepset/roberta-base-squad2` | ~125M | **CC-BY-4.0** | ✅ VERIFIED |
| Accuracy-max (GPU) `deepset/deberta-v3-large-squad2` | 434.0M | CC-BY-4.0 | ✅ VERIFIED |
| Light baseline `distilbert-base-cased-distilled-squad` | 65.2M | Apache-2.0 | ✅ VERIFIED |

**Decision:** `deepset/roberta-base-squad2` primary — SQuAD2-trained ⇒ **native no-answer via null-score threshold** (the "I don't know" mechanism), trains on H100, runs acceptably on CPU. CC-BY-4.0 = attribution required in product/docs. `deberta-v3-large-squad2` accuracy-max on GPU (**pin `transformers`/`tokenizers` — DeBERTa-v3 SentencePiece fast tokenizer historically brittle**). `distilbert-...-distilled-squad` is **SQuAD v1 → cannot natively abstain**; latency-floor baseline only.

### 2.4 Generative grounded reader

| HF ID | Params | License | Status |
|---|---|---|---|
| **PRIMARY** `google/flan-t5-base` | 247.6M | Apache-2.0 | ✅ VERIFIED |
| Quality upgrade `google/flan-t5-large` | 783.2M | Apache-2.0 | ✅ VERIFIED |
| No-train instruct LLM (optional) `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B | Apache-2.0 | ✅ VERIFIED |

**Decision:** `flan-t5-base` primary (Apache-2.0, CPU-runnable, trainable). Fine-tune on (question + retrieved context → answer) with explicit "answer 'I don't know' if context is insufficient" supervision. `flan-t5-large` is the fluency upgrade. `Qwen2.5-1.5B-Instruct` is the optional no-train abstractive reader (best zero-shot "IDK", but heavier CPU, needs chat-template + strict grounding prompt + citation enforcement).

### 2.5 Sparse baseline

- **BM25 via `rank_bm25` (BM25Okapi)** — primary sparse baseline (pip, pure-Python, no GPU/download; catches exact-term/entity matches dense misses). Not an HF model.
- **TF-IDF (`sklearn.TfidfVectorizer`)** — secondary sanity floor.
- **Hybrid:** fuse BM25 + dense (bge) via **Reciprocal Rank Fusion (RRF)** before reranking — reliably beats either alone on rare entities / OOD terms.

### 2.6 Recommended stack + chunking

```
RETRIEVE (top-50): rank_bm25 ⊕ BAAI/bge-base-en-v1.5  --RRF-->  (CPU fallback: all-MiniLM-L6-v2)
RERANK (top-5):    cross-encoder/ms-marco-MiniLM-L-6-v2   [swap → bge-reranker-v2-m3 on GPU]
READ (extractive): deepset/roberta-base-squad2 (null-score abstain)  [→ deberta-v3-large-squad2 GPU]
READ (generative): google/flan-t5-base (grounded + IDK)  [→ flan-t5-large / Qwen2.5-1.5B]
```

**Chunking notes:** `chunk_size` 512 tokens (~380 words), `chunk_overlap` 64 (12–15%), recursive splitter (`\n\n`→`\n`→sentence→token), min 64 tokens (merge forward). Per-chunk metadata: `doc_id, chunk_id, source, title, offset_start/end, hash, model_version, ingested_at`. Dedup by SHA-256 of normalized text ⇒ idempotent re-ingest. Note bge/all-MiniLM cap effective text ~384–512 tokens — chunk to fit the encoder.

---

## 3. PIPELINE + AGENT Architecture

A control-loop agent composing **query rewrite/decompose** (Rewrite-Retrieve-Read, 2305.14283), **Corrective-RAG** (2401.15884), and **Self-RAG** reflection (2310.11511). Default: fully local, CPU, deterministic heuristics at every node. Optional LLM brain at 3 nodes.

### 3.1 Tools (`kbqa.agent.tools`)

1. `analyze_query(q) -> {qtype, rewritten, sub_questions[{text, depends_on}]}` — rule-based default (multi-hop cues: `and / whose / that / before / most / compared to`, coreference, >1 NER; acronym + thesaurus rewrite). LLM mode: few-shot decomposition.
2. `retrieve(query, top_k=20, filters) -> list[Retrieved]` — FAISS dense (+optional BM25/RRF). `IndexFlatIP` <100k chunks, `IndexHNSWFlat(M=32)` ≥100k.
3. `rerank(query, cands, top_n=5) -> list[Retrieved]` — cross-encoder, sets `.rerank_score`.
4. `check_sufficiency(query, ctx) -> {verdict, score, missing_terms}` — heuristic: top rerank ≥ `TAU_HIGH`(0.55)→SUFFICIENT; `[TAU_LOW, TAU_HIGH)`→AMBIGUOUS; `<TAU_LOW`(0.15)→INSUFFICIENT; content-word coverage downgrades + reports `missing_terms`. LLM mode: Self-RAG ISREL.
5. `generate(query, ctx) -> {answer, citations[{source, chunk_id, span}]}` — extractive `roberta-base-squad2` default (span + source chunk); optional FLAN-T5 "answer using ONLY context; cite [chunk_id]".
6. `check_faithfulness(answer, ctx) -> {supported, support_score}` — NLI entailment (`cross-encoder/nli-deberta-v3-small` ⚠️ see note) between concatenated cited chunks (premise) and answer (hypothesis); `supported = P(entail) ≥ TAU_NLI`. Fallback: lexical/embedding overlap. LLM mode: Self-RAG ISSUP.
7. `make_clarifying_question(state) -> str` — when ambiguous after expansion.
8. *(pluggable)* `kg_query(logical_form) -> rows` — ChatKBQA-style NL→SPARQL backend; same AgentState, citations = triples.

> ⚠️ **Unverified ID:** `cross-encoder/nli-deberta-v3-small` (faithfulness NLI) was NOT verified in the research. **Verified fallback:** use the already-verified `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker score as a groundedness proxy, or pure embedding/lexical-overlap entailment using the verified `BAAI/bge-base-en-v1.5`. Verify the NLI model on the Hub before wiring it in; do not assert it exists.

### 3.2 AgentState (dataclass, threaded through the loop)

Key fields: `question`, `rewritten_query`, `qtype`(simple|multihop|unanswerable), `sub_questions:[SubQuestion]`, `contexts:[Retrieved]`, `sufficiency:{score,verdict,missing_terms}`, `answer`, `citations:[{source,chunk_id,span}]`, `faithful:bool`, `confidence`, control fields `iteration`/`max_iterations=3`/`needs_clarification`/`clarifying_question`, and `trace:[dict]` (audit log of every decision). `SubQuestion` carries `text, depends_on, answer, citations, contexts, status`(pending|answered|insufficient|failed). `Retrieved` carries `doc_id, chunk_id, text, source, dense_score, rerank_score`.

### 3.3 Three decision points (the agentic core)

1. **`analyze_query`** → route simple / multi-hop / unanswerable (unanswerable ⇒ immediate "I don't know").
2. **`check_sufficiency`** → SUFFICIENT (proceed) / AMBIGUOUS (rewrite+expand, widen top_k, retry) / INSUFFICIENT (decompose further or, if ambiguous after `max_iterations`, ask clarifying question). This is the **CRAG correction loop**, bounded by `max_iterations`.
3. **`check_faithfulness`** → emit cited answer if entailed, else **abstain "I don't know"** (Self-RAG ISSUP). A final global faithfulness gate runs on the synthesized multi-hop answer.

### 3.4 Control flow

Per sub-question: fill dependent slots from prior hops → retrieve → rerank → sufficiency (CRAG loop, widening top_k each retry) → generate → faithfulness (drop unsupported). Then **synthesize**: join answered sub-answers, dedup citations, `confidence = min` over hops, final faithfulness gate. Any required hop failing ⇒ abstain (optionally surface the supported partial answer). Every branch logged to `state.trace`.

### 3.5 Worked example (multi-hop, no paid LLM)

**Q:** "Which university did the founder of SpaceX attend, and in what year was that university established?"

- **Decompose** (cues `and`, two relations, dependent slots → multihop): SQ1 "Who founded SpaceX?" → SQ2(`depends_on=0`) "Which university did **{SQ1}** attend?" → SQ3(`depends_on=2`) "In what year was **{SQ2}** established?"
- **SQ1:** retrieve→rerank top "SpaceX was founded in 2002 by Elon Musk" (0.71 ≥ TAU_HIGH → SUFFICIENT) → extractive "Elon Musk" cite `spacex.md#c12` → NLI 0.94 supported ✔
- **SQ2** ("…did Elon Musk attend?"): iter0 score 0.34 → AMBIGUOUS (missing "university/attended"); `expand_query` adds {university, degree, graduated, alma mater}; iter1 → "Elon Musk transferred to the University of Pennsylvania…" 0.63 SUFFICIENT → "University of Pennsylvania" cite `bio.md#c07`, faithful 0.90 ✔
- **SQ3:** "…UPenn…was founded in 1740." 0.68 SUFFICIENT → "1740" cite `upenn.md#c03`, faithful 0.96 ✔
- **Synthesize + global gate:** "Elon Musk, the founder of SpaceX, attended the University of Pennsylvania, which was established in 1740." Citations `[spacex.md#c12, bio.md#c07, upenn.md#c03]`, confidence 0.90.
- **Abstention variant:** if UPenn founding date absent, SQ3 stays INSUFFICIENT after `max_iterations` ⇒ synthesis returns "I don't know" (or partial: "Elon Musk attended the University of Pennsylvania; I could not find its founding year in the provided documents"). **Never fabricates 1740 from parametric memory** — faithfulness gate requires entailment.

### 3.6 LLM brain plug-in points (no-LLM default vs upgrade)

| Node | No-LLM default (CPU) | Optional LLM upgrade |
|---|---|---|
| decompose | rule-based + thesaurus | few-shot decomposition |
| sufficiency | threshold + coverage | Self-RAG ISREL |
| generate | extractive span / FLAN-T5 | instruct LLM, cite chunk_ids |
| faithfulness | NLI / overlap | Self-RAG ISSUP |

---

## 4. DEPLOYMENT (FastAPI service)

Pipeline for `/ask`: embed query → FAISS ANN top-k → cross-encoder rerank top-n → reader (extractive span or FLAN-T5) → answer + citations + confidence + trace. Models pinned together by `MODEL_VERSION` env, held in in-process singletons + LRU caches.

### 4.1 FAISS vector store persistence

- **Build:** L2-normalize ⇒ cosine == inner product → `IndexFlatIP` (<100k chunks) or `IndexHNSWFlat(d, M=32)` (≥100k), wrapped in `IndexIDMap2` so FAISS ids == chunk row ids.
- **Persist:** `faiss.write_index(index, "kb.index")` + sidecar `meta.parquet` (id→text+metadata) + `manifest.json` (`{model_version, dim, metric, n_vectors, built_at}`).
- **Load:** `faiss.read_index(...)`; **assert `manifest.model_version == MODEL_VERSION`** (refuse cross-version index reuse — embeddings aren't cross-compatible).
- **Incremental ingest:** append-only `add_with_ids`; mirror rows to `meta.parquet`; deletes via tombstone set + `IDSelectorBatch` at query time; periodic offline `rebuild` compacts (HNSW adds but can't delete → mark-and-rebuild).

### 4.2 Endpoints

- `GET /health` → `{status, model_version, index:{loaded, n_vectors, metric}, uptime_s}`
- `POST /ingest` → chunk→embed→index; req `{documents[{doc_id,title,text,source,metadata}], chunking{size,overlap}, upsert}`; resp `{ingested_docs, new_chunks, skipped_duplicate_chunks, index_n_vectors, model_version, took_ms}`
- `POST /search` → retrieve+rerank; req `{query, top_k, rerank_top_n, filters, min_score}`; resp `{passages[{chunk_id,doc_id,title,text,source,offset,retriever_score,rerank_score,rank}], timing_ms, model_version}`
- `POST /ask` → full agentic RAG; req `{question, top_k, rerank_top_n, reader, max_new_tokens, require_citations, return_trace}`; resp `{answer, citations[{marker,chunk_id,doc_id,source,quote,offset}], confidence, is_answerable, trace{steps,model_version}, timing_ms}`. **Abstain:** top rerank < τ or extractive null-score wins ⇒ `is_answerable:false`, `answer:"I don't have enough information in the knowledge base."`
- `POST /batch` → `{requests[], reader, max_concurrency}` → `{results[], count, took_ms, model_version}`
- `GET /metrics` → Prometheus text (via `prometheus-fastapi-instrumentator`) + `?format=json` summary (`requests_total, p50/p95_ask_ms, cache_hit_rate, index_n_vectors, abstain_rate, model_version`)

**Confidence** = calibrated blend of top `rerank_score` + reader span/sequence probability + groundedness check.

### 4.3 Latency targets (CPU, single replica)

| Endpoint | p50 / p95 | Lever |
|---|---|---|
| `/health` | <5 ms | no model call |
| `/search` (k=20, 50→8) | 120 / 300 ms | batched encode, ONNX int8 bi-encoder, HNSW |
| `/ask` (extractive) | 350 / 800 ms | reuse `/search` + 1 reader pass |
| `/ask` (FLAN-T5-base) | 0.9 / 2.0 s | `max_new_tokens=256`, no bf16 on CPU |
| `/ingest` (per 1k chunks) | ~3–6 s | batched encode (batch 64), async |

CPU wins: ONNX Runtime + dynamic int8 on bi-encoder & cross-encoder (2–4×), tuned `OMP_NUM_THREADS`, query-embedding LRU cache, rerank only top-50.

### 4.4 Scalability, versioning, packaging

- Stateless API replicas behind LB; FAISS index + meta on shared read-only mmap volume (or per-replica copy); writes funneled to one ingest worker → publish new index version → replicas hot-swap.
- **Versioning:** `MODEL_VERSION` pins encoder+reranker+reader+index together; echoed in every response and `/metrics`; **blue/green index dirs** (`/kb/v1`, `/kb/v2`) for zero-downtime swap.
- **Gradio** demo (`demo.py`) calls `/ask`, shows answer + confidence + sources.
- **Docker:** `python:3.11-slim`, `pip install fastapi "uvicorn[standard]" gradio sentence-transformers==5.6.0 faiss-cpu pandas pyarrow prometheus-fastapi-instrumentator onnxruntime`; `ENV MODEL_VERSION OMP_NUM_THREADS=4`; expose 7860+8000.
- **HF Space:** SDK=`docker`, FAISS index via Git LFS or built at startup from a dataset repo, `MODEL_VERSION` as Space variable.

**Suggested layout:** `app.py`, `demo.py`, `Dockerfile`, `kb/{kb.index,meta.parquet,manifest.json}`, `kbqa/` package, `train/{retriever.py,reader_extractive.py,reader_flan_t5.py}`.

---

## 5. TRAINING Recipe (H100 80GB)

**Verified APIs (June 2026):** `sentence-transformers==5.6.0`, `transformers>=4.44`. Canonical imports: `from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments`; `from sentence_transformers.losses import ...`; `from sentence_transformers.util import mine_hard_negatives`. `predict_with_generate=True` for Seq2Seq; `doc_stride=128` + `version_2_with_negative` for SQuAD v2.

**Env:** `faiss-gpu` for mining, `faiss-cpu` for serving; enable TF32 (`torch.backends.cuda.matmul.allow_tf32=True`), bf16 for compute.

### 5.1 Retriever — MNRL + hard negatives (resume-safe)

1. **Mine hard negatives** with `mine_hard_negatives(pairs, model, num_negatives=8, range_min=10, range_max=60, max_score=0.85, margin=0.05, sampling_strategy="top", use_faiss=True, output_format="n-tuple")` — `range_min`/`max_score`/`margin` avoid false negatives.
2. **Train:** `CachedMultipleNegativesRankingLoss(model, mini_batch_size=64)` to push effective batch 256–1024 on H100 without OOM; `per_device_train_batch_size=256` (large batch = many in-batch negatives), `num_train_epochs=3`, `lr=2e-5`, `warmup_ratio=0.1`, cosine, `bf16=True`, `tf32=True`, `BatchSamplers.NO_DUPLICATES`, `weight_decay=0.01`, `load_best_model_at_end`, `metric_for_best_model="eval_cosine_ndcg@10"`.
3. **Eval:** `InformationRetrievalEvaluator(ndcg_at_k=[10], mrr_at_k=[10], accuracy_at_k=[1,5,10], precision_recall_at_k=[1,5,10])`.
4. **Iterate:** re-mine hard negatives with the fine-tuned model after round 1 (lifts Recall@k notably).
   Resume: `trainer.train(resume_from_checkpoint=True)`.

### 5.2 Extractive reader — EM/F1, doc_stride, no-answer (resume-safe)

- **Preprocess:** `MAX_LEN=384, DOC_STRIDE=128`, `truncation="only_second"`, `return_overflowing_tokens=True`, `return_offsets_mapping=True`; map answer span to token start/end; **no-answer (empty `answer_start`) ⇒ point start/end to `[CLS]`**; out-of-window spans ⇒ CLS too.
- **Train** `deepset/roberta-base-squad2`: `num_train_epochs=2`, `lr=3e-5`, `warmup_ratio=0.1`, `per_device_train_batch_size=48`, `bf16=True`, `tf32=True`, `weight_decay=0.01`, `load_best_model_at_end`, `metric_for_best_model="f1"`, `default_data_collator`.
- **Postprocess + score:** `evaluate.load("squad_v2")`; best (start,end) span vs **null score** (CLS logits); if `null_score - best_span_score > threshold` → no-answer (`version_2_with_negative=True`); tune `threshold` on dev. Report EM / F1 + **HasAns-F1 / NoAns-F1** separately.

### 5.3 (Optional) FLAN-T5 generative reader

- Prompt: "Answer the question using ONLY the context. If the answer is not in the context, say 'I don't know.'\nQuestion:{q}\nContext:{c}\nAnswer:"; input `max_length=1024`, target 128.
- `Seq2SeqTrainingArguments`: `num_train_epochs=3`, `lr=1e-4`, `per_device_train_batch_size=16`, `gradient_accumulation_steps=2`, **`bf16=True` (FLAN-T5 NaNs under fp16 — do NOT use fp16)**, `tf32=True`, `weight_decay=0.01`, `label_smoothing_factor=0.1`, **`predict_with_generate=True`**, `generation_max_length=128`, `metric_for_best_model="f1"`, `DataCollatorForSeq2Seq`.

### 5.4 Anti-overfitting (all stages)

`weight_decay=0.01`, 10% warmup, cosine/linear decay; early stopping via `load_best_model_at_end` + `EarlyStoppingCallback(patience=3)`; retriever `NO_DUPLICATES` + curated hard negs, 2–3 epochs; reader 2 epochs + label smoothing (FLAN-T5); hold out a true dev set **not** used in mining; watch train/eval divergence.

### 5.5 Resume-safe

All three stages use HF checkpointing: `trainer.train(resume_from_checkpoint=True)` (or explicit `checkpoint-XXXX`); `save_total_limit=3`, `save_steps` aligned with `eval_steps`.

### 5.6 Close train/serve gap

mine → train retriever → rebuild index → re-mine with new retriever → retrain (2 rounds) → **train reader on retrieved (not gold) contexts**.

---

## 6. METRICS + Baseline Plan

| Dimension | Metric | How |
|---|---|---|
| Retrieval | **Recall@{1,5,10}, NDCG@10, MRR@10** | `InformationRetrievalEvaluator` on dev qrels; use `rag-mini-bioasq` gold `relevant_passage_ids` or SQuAD-derived gold contexts |
| Answer quality | **EM / F1** | SQuAD v2 scorer (extractive) or normalized EM/F1 (generative) |
| Faithfulness / groundedness | % answer sentences entailed by cited passages | NLI model (verify ID) or LLM-judge over (answer sentence, cited chunk) |
| Citation accuracy | precision/recall of cited `chunk_id`s vs gold supporting chunks | set overlap on `[i]` markers → chunk_id |
| Abstention | no-answer precision/recall, abstain-rate | SQuAD v2 NoAns-F1; abstain-rate on unanswerables |
| Latency | p50/p95 per endpoint | `/metrics` histograms (§4.3) |

**Baseline plan (must beat).** Establish a floor before fine-tuning:
1. **Retrieval:** BM25 (`rank_bm25`) alone vs zero-shot `bge-base-en-v1.5` vs hybrid RRF vs **fine-tuned** retriever — track Recall@k/NDCG/MRR; fine-tuned + reranked must beat BM25.
2. **Reader:** zero-shot `roberta-base-squad2` vs **fine-tuned** (EM/F1 + NoAns-F1); FLAN-T5-base zero-shot vs fine-tuned (EM/F1 + faithfulness).
3. **End-to-end:** BM25 + zero-shot reader (no rerank, no agent loop) as the floor; full stack (hybrid retrieve → rerank → fine-tuned reader → agent loop) must improve EM/F1, faithfulness, and citation accuracy while keeping CPU latency within §4.3 targets.

---

## 7. Risks / Pitfalls / Fallbacks

| Risk | Mitigation / Verified fallback |
|---|---|
| **Dataset unavailability / license** (`trivia_qa` unknown; MS MARCO research-terms) | Drop to `neural-bridge/rag-dataset-12000` (Apache-2.0) for generative; `Tevatron/msmarco-passage` (Apache-2.0) for retriever; **B4/C4 SQuAD-derived** pairs+KB need zero extra download and keep train/serve distribution aligned. Legal sign-off before any commercial use of flagged sets. |
| **Hallucination** | Grounded reader + `check_faithfulness` entailment gate ⇒ **abstain "I don't know"** rather than fabricate; extractive null-score abstention; final global faithfulness gate on synthesized answers. Never answer from parametric memory. |
| **Long documents** | 512-token chunks, 64 overlap, recursive splitter, `doc_stride=128` sliding window in the reader; gte-base-en-v1.5 (8192 ctx) only if truly needed (accept `trust_remote_code` cost). |
| **CPU latency** | ONNX int8 quantization (2–4×), HNSW ANN, rerank only top-50, MiniLM CPU-fallback encoder (¼ index), LRU caches, `max_new_tokens=256`; keep `bge-reranker-v2-m3`/`deberta-v3-large`/`flan-t5-large` behind GPU config flags. |
| **Multi-hop precision loss vs SPARQL** | Recover via agent decompose + iterative retrieval + sufficiency check; offer pluggable `kg_query` (ChatKBQA-style) backend for customers with a real KG. |
| **Cross-version index reuse** | `manifest.model_version` assertion on load; blue/green index dirs; re-index on any encoder swap (embeddings not cross-compatible). |
| **Tokenizer breakage** (DeBERTa-v3) | Pin `transformers`/`tokenizers`; prefer `roberta-base-squad2` default. |
| **FLAN-T5 fp16 NaN** | Train bf16 only; never fp16. |
| **Unverified NLI faithfulness model** (`cross-encoder/nli-deberta-v3-small`) | ⚠️ Not verified in research. Verify on Hub before use; fallback to verified `bge-base-en-v1.5` embedding-overlap or `ms-marco-MiniLM-L-6-v2` rerank-score groundedness proxy. |

---

### Verified-ID quick reference (assert only these)

**Datasets (all ✅):** `rajpurkar/squad_v2`, `hotpotqa/hotpot_qa`, `mandarjoshi/trivia_qa`(license unknown ⚠️), `sentence-transformers/natural-questions`, `sentence-transformers/msmarco`(MS MARCO terms ⚠️), `Tevatron/msmarco-passage`, `rag-datasets/rag-mini-wikipedia`, `rag-datasets/rag-mini-bioasq`, `neural-bridge/rag-dataset-12000`, `BeIR/scifact`.
**Models (all ✅):** `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2`, `sentence-transformers/all-MiniLM-L6-v2`, `Alibaba-NLP/gte-base-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-v2-m3`, `deepset/roberta-base-squad2`, `deepset/deberta-v3-large-squad2`, `distilbert-base-cased-distilled-squad`, `google/flan-t5-base`, `google/flan-t5-large`, `Qwen/Qwen2.5-1.5B-Instruct`. Plus `rank_bm25` (pip).
**Unverified — flagged, do not assert:** `cross-encoder/nli-deberta-v3-small` (faithfulness NLI) — use verified fallback above.