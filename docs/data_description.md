# Data Description Document

**Project #3 — Knowledge Base Question-Answering System (`kbqa`)**
**Author:** Le Dinh Minh Quan (23127460)
**Approach:** Agentic RAG-over-documents (FAISS hybrid retrieval + cross-encoder rerank + CRAG/Self-RAG loop + grounded, cited generation with safe abstention).
**Scope of this document:** Assignment §4 — data sources, licenses, sizes, language, preprocessing, splits, schemas, and known limitations. Every dataset ID, config, split, row count, and license below is taken verbatim from the verified `DESIGN_BRIEF.md` (HF dataset-viewer confirmation, 2026-06-26). No fact here is invented.

---

## 1. Overview

The `kbqa` system is trained and evaluated on a layered data stack that separates four concerns: (a) **reader supervision** (span extraction + learning to abstain), (b) **retriever supervision** (question→passage relevance), (c) **a self-contained demo/eval knowledge base** (so the whole pipeline is reproducible end-to-end on one machine), and (d) **retrieval-recall ground truth** (gold passage IDs). All corpora are **English**. No large data is committed to the repository — everything is pulled on demand by `scripts/download_data.py` (see §7).

```
 reader QA ──────►  span + no-answer supervision   (SQuAD v2, HotpotQA, [TriviaQA opt.])
 retriever pairs ─►  question↔passage relevance     (NQ pairs, [MS MARCO opt.], Tevatron)
 demo KB ────────►  end-to-end reproducible corpus  (rag-mini-wikipedia)
 recall qrels ───►  gold passage IDs for Recall@k   (rag-mini-bioasq)
 generative RAG ─►  grounded answer + IDK fine-tune  (neural-bridge/rag-dataset-12000)
```

---

## 2. Data Sources, HF IDs, and Licenses

### 2.1 Reader QA datasets (question, context, answer)

| Role | HF ID | Config / Split (rows) | License |
|---|---|---|---|
| **PRIMARY** extractive + abstain | `rajpurkar/squad_v2` | `squad_v2` / train 130,319 · val 11,873 | CC-BY-SA-4.0 |
| Multi-hop / agentic eval | `hotpotqa/hotpot_qa` | `distractor` / train 90,447 · val 7,405 | CC-BY-SA-4.0 |
| Generative (optional) | `mandarjoshi/trivia_qa` | `rc.nocontext` / train 138,384 | **unknown** ⚠️ |

**SQuAD v2** is the reader backbone: its ~50K **unanswerable** questions are the single most valuable asset for production-grade abstention — the reader is trained to point at the null/`[CLS]` span instead of hallucinating. **HotpotQA-distractor** supplies multi-hop questions for the agentic decomposition path. **TriviaQA** is **optional only**: its HF license is `unknown` (research-friendly but not an SPDX tag), so it ⚠️ **must not be used for commercial training without legal sign-off**.

### 2.2 Retriever pair datasets (question, positive passage)

| Role | HF ID | Config / Split (rows) | License |
|---|---|---|---|
| **PRIMARY** fast start | `sentence-transformers/natural-questions` | `pair` / train 100,231 | CC-BY-SA-3.0 |
| Scale (triplets, IDs) | `sentence-transformers/msmarco` | `triplets` / 397.2M | **MS MARCO research terms** ⚠️ |
| Text inlined (hard negs) | `Tevatron/msmarco-passage` | `default` / train 400,782 · val 6,980 | Apache-2.0 |

The first bi-encoder fine-tune uses `sentence-transformers/natural-questions` (`pair`) — clean `query`/`answer` pairs, MNRL-ready, no joins. Scaling to `sentence-transformers/msmarco` (`triplets`) yields stronger general retrieval but carries ⚠️ **MS MARCO research terms** (not an SPDX license — flag for legal before any commercial use). `Tevatron/msmarco-passage` is the Apache-2.0, text-inlined alternative with pre-mined hard negatives. A **zero-download fallback** derives retriever positives directly from SQuAD v2 (each answerable `(question, context)` is a positive pair), keeping retriever and reader on the same distribution.

### 2.3 Demo / eval RAG corpus + QA (self-contained KB)

| Role | HF ID | Configs (rows) | License |
|---|---|---|---|
| **PRIMARY** demo KB | `rag-datasets/rag-mini-wikipedia` | `text-corpus`/passages 3,200 · `question-answer`/test 918 | CC-BY-3.0 |
| + retrieval-recall (gold IDs) | `rag-datasets/rag-mini-bioasq` | `text-corpus`/passages 40,200 · `question-answer-passages`/test 4,719 | CC-BY-2.5 |
| Generative-RAG fine-tune | `neural-bridge/rag-dataset-12000` | `default` train 9,600 / test 2,400 | **Apache-2.0** ✅ |
| IR benchmark (optional) | `BeIR/scifact` | corpus 5,183 · queries 1,109 (+`BeIR/scifact-qrels`) | CC-BY-SA-4.0 |

`rag-mini-wikipedia` is the default reproducible KB: index the `text-corpus`/`passages` config (3,200 passages), evaluate on `question-answer`/`test` (918 QA). Note its QA rows carry **no gold passage IDs**, so it measures **end-to-end answer quality only**. For **retrieval Recall@k / MRR**, `rag-mini-bioasq` is used because its `question-answer-passages` rows include `relevant_passage_ids`. `neural-bridge/rag-dataset-12000` is the cleanest-license (Apache-2.0) generative-RAG fine-tune set.

### 2.4 License summary (production posture)

| Tier | Datasets |
|---|---|
| **Cleanest commercial (Apache-2.0)** | `neural-bridge/rag-dataset-12000`, `Tevatron/msmarco-passage` (repo tag — derived from MS MARCO data; verify upstream) |
| **CC-BY-SA / CC-BY (attribution ± share-alike)** | `rajpurkar/squad_v2`, `hotpotqa/hotpot_qa`, `sentence-transformers/natural-questions`, `rag-datasets/rag-mini-wikipedia`, `rag-datasets/rag-mini-bioasq`, `BeIR/scifact` |
| ⚠️ **Flag for legal review** | `mandarjoshi/trivia_qa` (**unknown**), MS MARCO family — `sentence-transformers/msmarco` (**MS MARCO research terms**) |

> **Attribution requirement:** the CC-BY-SA / CC-BY corpora require attribution (and, for SA variants, share-alike on derivatives) in the product and documentation. The two ⚠️ flagged items are restricted to research/optional use until legal sign-off.

---

## 3. Sizes and Language

- **Language:** all corpora are **English** only. There is no multilingual coverage by design (see limitations, §6).
- **Largest reader corpus:** SQuAD v2 (130,319 train + 11,873 val).
- **Largest retriever source:** `sentence-transformers/msmarco` `triplets` (397.2M triplets — used at scale only; primary fine-tune stays on the 100,231-pair NQ set).
- **Demo KB footprint:** intentionally small — 3,200 passages (rag-mini-wikipedia) — so the entire ingest→index→ask loop runs reproducibly on CPU.
- **Recall qrels:** rag-mini-bioasq, 40,200 passages + 4,719 QA rows with gold passage IDs.

---

## 4. Preprocessing

All preprocessing is deterministic and idempotent so that re-ingesting the same source produces the same index.

| Step | Specification |
|---|---|
| **Text cleaning** | Normalize whitespace and unicode; strip boilerplate/markup from PDF/MD/HTML/text on ingest; preserve sentence boundaries for the recursive splitter. |
| **Chunking** | `chunk_size = 512 tokens` (~380 words), `chunk_overlap = 64 tokens` (12–15%), **recursive splitter** (`\n\n` → `\n` → sentence → token); minimum 64 tokens (merge forward). Chunked to fit the encoder's effective window (bge / all-MiniLM cap ≈ 384–512 tokens). |
| **Deduplication** | **SHA-256 hash of normalized chunk text**; duplicate chunks are skipped on ingest → idempotent re-ingest (`skipped_duplicate_chunks` reported by `/ingest`). |
| **SQuAD v2 no-answer handling** | Empty `answers.text` / `answer_start` ⇒ **unanswerable**. During reader preprocessing the answer span maps to `[CLS]` (null span); spans outside the `doc_stride=128` sliding window also map to `[CLS]`. This is the supervision signal for the "I don't know" mechanism. |
| **Query / passage prefixes** | Encoder-specific and **mandatory** (wrong prefixes silently degrade recall): **bge-base-en-v1.5** prepends `"Represent this sentence for searching relevant passages: "` to **queries only** (passages raw); **e5-base-v2** prepends `"query: "` / `"passage: "`; **all-MiniLM** is symmetric — **no prefix**. All embeddings are L2-normalized (cosine ≡ inner product). |
| **Chunk metadata** | Each chunk stores `doc_id, chunk_id, source, title, offset_start/end, hash, model_version, ingested_at` for citation and audit. |
| **Gold-ID parsing** | rag-mini-bioasq `relevant_passage_ids` arrives as a string (e.g. `"[9797, 11906]"`) and is parsed with `ast.literal_eval`. |

---

## 5. Train / Validation / Test Splits and Justification

| Dataset | Split policy | Justification |
|---|---|---|
| `rajpurkar/squad_v2` | **Official** train 130,319 / val 11,873; carve a held-out dev slice from train for the **retriever** and for the no-answer **threshold** tuning. | Using the official splits keeps EM/F1 + NoAns-F1 comparable to published numbers; a separate dev slice prevents the retriever (and the null-score threshold) from being tuned on the reader's evaluation set. |
| `hotpotqa/hotpot_qa` (`distractor`) | Official train 90,447 / val 7,405; used as **eval** for multi-hop decomposition. | The distractor setting tests retrieval + reasoning jointly, matching the agent's decompose→retrieve→synthesize path. |
| Retriever pairs (NQ / MS MARCO / Tevatron) | Train on pairs/triplets; hold out a **true dev set not used in hard-negative mining**. | Hard negatives mined from the training pool can leak into eval; a clean dev set keeps `InformationRetrievalEvaluator` (NDCG@10 / MRR@10 / Recall@{1,5,10}) honest. |
| `rag-datasets/rag-mini-wikipedia` | `text-corpus`/passages indexed; `question-answer`/test (918) for **end-to-end** answer quality. | QA rows have **no gold passage IDs** → suitable for answer EM/F1 and faithfulness, **not** retrieval recall. |
| `rag-datasets/rag-mini-bioasq` | `question-answer-passages`/test (4,719) with **gold `relevant_passage_ids`** → **retrieval Recall@k / MRR**. | The only corpus with gold passage IDs; this is where retrieval-recall is measured exactly. |
| `neural-bridge/rag-dataset-12000` | `default` train 9,600 / test 2,400. | Pre-split generative-RAG set (Apache-2.0) for the grounded FLAN-T5 fine-tune (answer + "I don't know" supervision). |

> **SQuAD-derived fallback:** dedup SQuAD v2 `context` into a ~1.2K-passage KB where each QA item has a known gold context — enabling **exact retrieval-recall measurement with zero extra download** and a perfectly aligned train/serve distribution.

---

## 6. Known Limitations and Biases

- **Wikipedia / Western-centric content.** SQuAD, NQ, HotpotQA, and rag-mini-wikipedia derive from English Wikipedia and web text → topical and cultural skew toward Western, well-documented entities; long-tail and non-Western topics are under-represented.
- **Domain shift.** The demo KB (general Wikipedia) and the biomedical recall set (rag-mini-bioasq) differ in domain. A retriever/reader tuned on open-domain QA will degrade on specialized corpora; this is why retrieval recall is reported separately on bioasq and why the system supports re-indexing on customer documents.
- **English-only.** No multilingual data → the system is not validated outside English; cross-lingual queries are out of scope.
- **License-restricted sources.** ⚠️ `trivia_qa` (unknown) and MS MARCO family (research terms) cannot be relied on for commercial deployment without legal review — production training should prefer the Apache-2.0 / CC-BY tiers.
- **No gold passage IDs in the demo KB.** rag-mini-wikipedia QA cannot validate retrieval ranking; recall claims must cite rag-mini-bioasq (or the SQuAD-derived KB), not the demo set.
- **Span/short-answer bias.** SQuAD/NQ favor short extractive answers, biasing the reader toward span extraction over long-form synthesis; the generative reader and multi-hop synthesis partially offset this.

---

## 7. Download Script and Data Hygiene

No large data is committed. `scripts/download_data.py` exposes `--reader`, `--retriever`, and `--demo-kb` flags, each calling `datasets.load_dataset(...)` with the **exact verified config/split** and writing only to the HF cache. `data/.gitignore` blocks `*.parquet` / `*.arrow`. A `make data-demo` target materializes only the small `rag-mini-wikipedia` corpus and builds the demo FAISS index — keeping the repository lightweight and reproducible.

---

## 8. Per-Dataset Schema Reference

**`rajpurkar/squad_v2`** (`squad_v2`)

| Field | Type | Notes |
|---|---|---|
| `id` | str | unique question id |
| `title` | str | article title |
| `context` | str | passage |
| `question` | str | natural-language question |
| `answers` | {`text`: list[str], `answer_start`: list[int32]} | **empty lists ⇒ unanswerable** |

**`hotpotqa/hotpot_qa`** (`distractor`)

| Field | Type | Notes |
|---|---|---|
| `id` | str | question id |
| `question` | str | multi-hop question |
| `answer` | str | gold answer |
| `context` | {title, sentences} | candidate paragraphs (incl. distractors) |
| `supporting_facts` | {title, sent_id} | gold supporting sentences |

**`sentence-transformers/natural-questions`** (`pair`)

| Field | Type | Notes |
|---|---|---|
| `query` | str | question |
| `answer` | str | positive passage (MNRL positive) |

**`Tevatron/msmarco-passage`** (`default`)

| Field | Type | Notes |
|---|---|---|
| `query_id` / `query` | str | question id + text |
| `positive_passages` | list[{docid, title, text}] | gold passages |
| `negative_passages` | list[{docid, title, text}] | pre-mined hard negatives |

**`rag-datasets/rag-mini-wikipedia`**

| Config | Field | Type | Notes |
|---|---|---|---|
| `text-corpus` | `passage`, `id` | str | 3,200 KB passages to index |
| `question-answer` | `question`, `answer` | str | 918 QA; **no gold passage IDs** |

**`rag-datasets/rag-mini-bioasq`**

| Config | Field | Type | Notes |
|---|---|---|---|
| `text-corpus` | `passage`, `id` | str | 40,200 passages |
| `question-answer-passages` | `question`, `answer`, `relevant_passage_ids` | str | gold IDs as `"[9797, 11906]"` → `ast.literal_eval` |

**`neural-bridge/rag-dataset-12000`** (`default`)

| Field | Type | Notes |
|---|---|---|
| `context` | str | retrieved/grounding passage |
| `question` | str | question |
| `answer` | str | grounded answer (incl. IDK supervision) |

---

*All dataset IDs, configs, splits, row counts, and licenses in this document are verified against the project `DESIGN_BRIEF.md` (HF dataset-viewer, 2026-06-26). Items marked ⚠️ require legal review before commercial use; items marked ✅ are the cleanest-license choices.*
