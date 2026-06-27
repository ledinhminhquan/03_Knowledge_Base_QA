# Model Card (MODEL_CARD)

**System:** `kbqa` — Knowledge Base Question-Answering System (Project #3, NLP-in-Industry)
**Author:** Le Dinh Minh Quan (student 23127460)
**Type:** Composite RAG-over-documents system (retriever + reranker + reader), not a single weights checkpoint.
**Date:** 2026-06-26 · **License of the composite:** see per-component table (§ Model Details). Most permissive overall constraint is **CC-BY-4.0 attribution** (inherited from the extractive reader `deepset/roberta-base-squad2`).

> This card follows the Hugging Face model-card structure. `kbqa` is a *pipeline*: a dense bi-encoder retriever, a cross-encoder reranker, and an extractive (or generative) reader, orchestrated by a deterministic agentic state machine (CRAG correction loop + Self-RAG reflection). It produces a **grounded, cited answer with a confidence score**, or it **abstains** with "I don't know." It runs **on CPU by default**; GPU is a speed/accuracy knob, not a requirement. No paid API is ever required.

---

## Model Details

### System summary

```
ingest/chunk/index (FAISS) -> analyze/route query -> retrieve (BM25 + bge dense, RRF)
  -> rerank (cross-encoder) -> sufficiency check (CRAG loop, max_iterations=3)
  -> grounded generate w/ citations -> faithfulness gate
  -> answer + citations + confidence   |   abstain "I don't know"
```

### Component models, versions, and licenses

All IDs verified live against the Hugging Face Hub (2026-06-26). The pipeline is pinned together by a single `MODEL_VERSION` environment variable, echoed in every API response and in `/metrics`.

| Role | HF ID | Params | License | Notes |
|---|---|---|---|---|
| **Retriever (primary)** | `BAAI/bge-base-en-v1.5` | 109.5M | MIT | 768-dim; query-prefix required (below) |
| Retriever (CPU fallback) | `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | Apache-2.0 | 384-dim, symmetric (no prefix); re-index on swap |
| **Reranker (CPU default)** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22.7M | Apache-2.0 | top-50 → top-5; single-digit ms/pair CPU |
| Reranker (GPU upgrade) | `BAAI/bge-reranker-v2-m3` | 567.8M | Apache-2.0 | config-swappable; ~25× CPU cost |
| **Reader — extractive (primary)** | `deepset/roberta-base-squad2` | ~125M | **CC-BY-4.0** | null-score abstain; **attribution required** |
| Reader — extractive (GPU max) | `deepset/deberta-v3-large-squad2` | 434.0M | CC-BY-4.0 | pin `transformers`/`tokenizers` (SP fast tokenizer brittle) |
| **Reader — generative (primary)** | `google/flan-t5-base` | 247.6M | Apache-2.0 | grounded + "I don't know" supervision |
| Reader — generative (upgrade) | `google/flan-t5-large` | 783.2M | Apache-2.0 | fluency upgrade (GPU) |
| Optional no-train LLM brain | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B | Apache-2.0 | optional abstractive reader / agent brain |
| Sparse baseline | `rank_bm25` (BM25Okapi) | — | pip pkg | not an HF model; hybrid via RRF |

**Required attribution (CC-BY-4.0).** The extractive reader `deepset/roberta-base-squad2` is licensed **CC-BY-4.0**; any product or documentation shipping this default reader **must attribute deepset** and reproduce the license notice. `deepset/deberta-v3-large-squad2` carries the same CC-BY-4.0 obligation. SQuAD v2 training data is CC-BY-SA-4.0 (attribution + share-alike on derived datasets).

**Query/passage prefix conventions (getting these wrong silently degrades recall):**
- **bge-base-en-v1.5:** prepend `"Represent this sentence for searching relevant passages: "` to **queries only**; passages embedded raw.
- **all-MiniLM-L6-v2:** symmetric — **no prefixes**.
- All encoders: L2-normalize embeddings; cosine == inner product after normalization.

**Compatibility constraint.** Embeddings are **not** cross-compatible across encoders. The FAISS `manifest.json` records `model_version`; on load the service **asserts `manifest.model_version == MODEL_VERSION`** and refuses cross-version index reuse. Switching the retriever requires a full re-index.

### Architecture

A deterministic agentic state machine with **three decision points**:
1. **analyze/route** — classify simple / multi-hop / unanswerable (unanswerable ⇒ immediate "I don't know").
2. **sufficiency loop (CRAG)** — top rerank ≥ `TAU_HIGH=0.55` ⇒ SUFFICIENT; `[TAU_LOW=0.15, 0.55)` ⇒ AMBIGUOUS (rewrite/expand, widen top_k, retry); `< 0.15` ⇒ INSUFFICIENT. Bounded by `max_iterations=3`.
3. **faithfulness gate (Self-RAG ISSUP)** — emit cited answer only if entailed by retrieved context; else **abstain**.

Every node has a no-LLM CPU heuristic default; an **optional local LLM brain** upgrades decompose, sufficiency, and faithfulness. A pluggable `kg_query` backend (ChatKBQA-style NL→SPARQL over a knowledge graph) sits beside `retrieve` behind the same agent interface but is **not** the default approach.

---

## Intended Use

**Primary intended use.** Grounded, cited question answering over a *user-supplied document corpus* (PDF/MD/HTML/text ingested, chunked at 512 tokens / 64 overlap, indexed in FAISS). The system answers natural-language questions with answers traceable to source spans, returns a calibrated confidence, and **abstains** when the corpus lacks support. Target users: developers building an internal/enterprise knowledge assistant who need auditable citations and safe refusal rather than fluent fabrication.

**In-scope tasks.** Single-hop factual QA; multi-hop QA via agent decomposition + iterative retrieval; retrieval/search-only via `/search`; batch QA via `/batch`. English-language documents.

**Out-of-scope / misuse (do NOT use for):**
- **Open-domain QA without an indexed corpus** — the system answers *only* from retrieved context; with no relevant chunks it abstains by design.
- **Non-English content** — all components are English-only; non-English queries/documents are unsupported and degrade silently.
- **High-stakes autonomous decisions** (medical, legal, financial, safety) without human review. Even with the faithfulness gate, retrieval and entailment are imperfect.
- **Authoritative reasoning beyond the documents** — no use of parametric/world knowledge is permitted; the system is not a general chatbot.
- **Adversarial / untrusted document ingestion** without sanitization — see prompt-injection caveat below.
- **Commercial deployment of license-flagged training data** (e.g., `mandarjoshi/trivia_qa` unknown license; MS MARCO research terms) without legal sign-off.

---

## Training Data

Components are used zero-shot or fine-tuned on the datasets below; no large data is committed to the repo (`scripts/download_data.py` pulls on demand). All IDs verified live on the HF dataset-viewer (2026-06-26).

| Stage | Dataset | Config / rows | License |
|---|---|---|---|
| Extractive reader (backbone + abstention) | `rajpurkar/squad_v2` | train 130,319 · val 11,873 (~50K unanswerable) | CC-BY-SA-4.0 |
| Multi-hop eval | `hotpotqa/hotpot_qa` | distractor: train 90,447 · val 7,405 | CC-BY-SA-4.0 |
| Retriever pairs (primary) | `sentence-transformers/natural-questions` | pair: train 100,231 | CC-BY-SA-3.0 |
| Generative-RAG fine-tune | `neural-bridge/rag-dataset-12000` | train 9,600 / test 2,400 | **Apache-2.0** |
| Demo KB (reproducible) | `rag-datasets/rag-mini-wikipedia` | text-corpus 3,200 passages · QA 918 | CC-BY-3.0 |
| Retrieval-recall (gold IDs) | `rag-datasets/rag-mini-bioasq` | corpus 40,200 · QA-passages 4,719 | CC-BY-2.5 |

**Why SQuAD v2 is central.** Its ~50K **unanswerable** questions are the key asset for production-grade abstention: the extractive reader learns to point at `[CLS]`/null instead of hallucinating, exposed at inference as the **null-score threshold** ("I don't know" mechanism).

**Training recipe (H100 80GB, summary).** Retriever: hard-negative mining + `CachedMultipleNegativesRankingLoss`, effective batch 256–1024, 3 epochs, lr 2e-5, bf16+tf32, re-mine after round 1. Extractive reader: `MAX_LEN=384`, `DOC_STRIDE=128`, no-answer span → `[CLS]`, 2 epochs, lr 3e-5, `version_2_with_negative=True`, tune null threshold on dev. FLAN-T5: grounded "answer using ONLY context; else 'I don't know'" prompt, **bf16 only (fp16 NaNs)**, `predict_with_generate=True`. Anti-overfitting: `weight_decay=0.01`, 10% warmup, cosine decay, `EarlyStoppingCallback(patience=3)`, held-out dev not used in mining.

---

## Evaluation

> **All numbers below are PROJECTED targets for this assignment, clearly marked. They are not measured benchmark results.** Final measured values populate the evaluation report after training.

### Metrics

| Dimension | Metric | Source |
|---|---|---|
| Retrieval | Recall@{1,5,10}, NDCG@10, MRR@10 | `InformationRetrievalEvaluator`; `rag-mini-bioasq` gold `relevant_passage_ids` or SQuAD-derived gold contexts |
| Answer quality | EM / F1 | SQuAD v2 scorer (extractive) / normalized EM/F1 (generative) |
| Abstention | NoAns-F1, abstain-rate, no-answer precision/recall | SQuAD v2 NoAns split; unanswerables |
| Faithfulness | % answer sentences entailed by cited passages | NLI / LLM-judge over (sentence, cited chunk) |
| Citation accuracy | precision/recall of cited `chunk_id`s vs gold | set overlap on `[i]` markers → chunk_id |
| Latency | p50/p95 per endpoint | `/metrics` histograms |

### Projected numbers (PROJECTED — illustrative targets, not measured)

| Metric | Baseline (BM25 + zero-shot reader) | Full stack (PROJECTED target) |
|---|---|---|
| Retrieval Recall@5 | ~0.62 (PROJECTED) | ~0.85 (PROJECTED) |
| Retrieval NDCG@10 | ~0.55 (PROJECTED) | ~0.78 (PROJECTED) |
| Reader F1 (HasAns) | ~0.68 (PROJECTED) | ~0.83 (PROJECTED) |
| NoAns-F1 (abstention) | ~0.70 (PROJECTED) | ~0.86 (PROJECTED) |
| Faithfulness (entailed) | n/a | ≥ 0.90 (PROJECTED) |

**Baseline plan (must beat).** Floor = BM25 (`rank_bm25`) + zero-shot reader, no rerank, no agent loop. The full stack (hybrid BM25⊕bge → RRF → cross-encoder rerank → fine-tuned reader → agent loop) must improve EM/F1, faithfulness, and citation accuracy while staying within CPU latency targets.

### Latency (CPU, single replica — projected targets)

| Endpoint | p50 / p95 |
|---|---|
| `/health` | < 5 ms |
| `/search` (k=20, 50→8) | 120 / 300 ms |
| **`/ask` (extractive)** | **350 / 800 ms** |
| `/ask` (FLAN-T5-base) | 0.9 / 2.0 s |

CPU levers: ONNX Runtime + dynamic int8 on bi-encoder & cross-encoder (2–4×), HNSW ANN, rerank only top-50, query-embedding LRU cache, tuned `OMP_NUM_THREADS`.

---

## Ethical Considerations

- **Hallucination control by design.** The grounded reader plus the `check_faithfulness` entailment gate cause the system to **abstain** ("I don't know") rather than fabricate. Extractive null-score abstention and a final global faithfulness gate on synthesized multi-hop answers reinforce this. The system **never answers from parametric memory**.
- **Provenance and auditability.** Every answer ships with citations (`source`, `chunk_id`, span/quote) and a full decision `trace`, so a human can verify each claim against its source.
- **Bias inheritance.** Answers reflect the ingested corpus and the biases of the pretraining/fine-tuning data (Wikipedia, NQ, SQuAD, MS-MARCO-derived). The system surfaces what the documents say, including any errors or bias therein; it does not correct them.
- **License compliance.** CC-BY-4.0 attribution for the reader, CC-BY-SA share-alike for derived datasets, and the legal-flag on `trivia_qa` / MS MARCO must be respected before any commercial use.

---

## Caveats and Limitations

- **Abstention is expected behaviour, not failure.** When top rerank < `TAU_LOW` or the extractive null-score wins, the API returns `is_answerable: false` and `"I don't have enough information in the knowledge base."` Tune `TAU_HIGH=0.55` / `TAU_LOW=0.15` per corpus.
- **English-only.** All retriever, reranker, and reader components are English; other languages are unsupported.
- **Prompt-injection / poisoned documents.** Because answers are grounded in ingested text, a malicious document can attempt to steer a generative reader. Mitigations: prefer the **extractive** reader (returns spans, not free generation), enforce citation requirements, sanitize/trust-scope ingestion, keep the faithfulness gate on. Do not ingest untrusted documents into a trusted KB without review.
- **Multi-hop precision** is recovered via decomposition + iterative retrieval but does not match exact SPARQL joins; customers with a real knowledge graph can enable the pluggable `kg_query` backend.
- **Cross-version index reuse is blocked** via the `manifest.model_version` assertion; re-index on any encoder swap.
- **Tokenizer / dtype pitfalls:** pin `transformers`/`tokenizers` for DeBERTa-v3; train FLAN-T5 in bf16 only (fp16 NaNs).

---

## How to Load / Serve

**Serving (FastAPI + Gradio).** The service exposes `GET /health`, `POST /ingest`, `POST /search`, `POST /ask`, `POST /batch`, `GET /metrics` (Prometheus + `?format=json`), with a Gradio demo on port **7860**. Models are held in in-process singletons + LRU caches, pinned together by `MODEL_VERSION`.

```bash
# Docker (python:3.11-slim)
docker build -t kbqa .
docker run -e MODEL_VERSION=v1 -e OMP_NUM_THREADS=4 -p 7860:7860 -p 8000:8000 kbqa
# -> http://localhost:7860 (Gradio)  ·  http://localhost:8000/ask (API)
```

```python
# Load components directly
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import faiss

retriever = SentenceTransformer("BAAI/bge-base-en-v1.5")      # MIT
reranker  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Apache-2.0
reader    = pipeline("question-answering",
                     model="deepset/roberta-base-squad2")     # CC-BY-4.0 — attribute deepset

index = faiss.read_index("kb/kb.index")
# assert manifest.model_version == MODEL_VERSION before serving

# bge: prefix QUERIES only
PREFIX = "Represent this sentence for searching relevant passages: "
q_emb = retriever.encode([PREFIX + "When was UPenn founded?"], normalize_embeddings=True)
```

**FAISS persistence:** `kb/kb.index` + `meta.parquet` (id→text+metadata) + `manifest.json` (`{model_version, dim, metric, n_vectors, built_at}`). HF Space: SDK `docker`, `MODEL_VERSION` as a Space variable, port 7860.

---

### Verified-ID reference (assert only these for models)

`BAAI/bge-base-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-v2-m3`, `deepset/roberta-base-squad2` (CC-BY-4.0), `deepset/deberta-v3-large-squad2` (CC-BY-4.0), `google/flan-t5-base`, `google/flan-t5-large`, `Qwen/Qwen2.5-1.5B-Instruct`, plus `rank_bm25` (pip). The faithfulness NLI model is intentionally unspecified here (`cross-encoder/nli-deberta-v3-small` is **unverified**); use a verified `bge`/`ms-marco` groundedness fallback until confirmed on the Hub.

---

*This card describes a student project (Project #3, NLP-in-Industry). Numbers marked PROJECTED are assignment targets, not measured results.*
