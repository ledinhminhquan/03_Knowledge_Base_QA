# Dataset Card (DATA_CARD)

**Project:** Knowledge Base Question-Answering System (`kbqa`) — Project #3, NLP-in-Industry
**Author:** Le Dinh Minh Quan (student 23127460)
**Document status:** All dataset IDs, configs, splits, row counts, and licenses below are **VERIFIED live** against the Hugging Face Hub dataset viewer (2026-06-26) unless explicitly flagged. This card consolidates the five datasets that the `kbqa` system actually depends on; it is the authoritative data-governance reference for the report and the repository.

---

## 1. Summary

`kbqa` is a production RAG-over-documents question-answering system. It does **not** train its own corpus from scratch; instead it composes five public Hugging Face datasets, each serving a distinct role across reader supervision, retriever supervision, the self-contained demo knowledge base, retrieval-recall benchmarking, and generative-RAG fine-tuning. This card documents intended use, composition, collection, licensing (with the attribution and share-alike obligations that CC-BY / CC-BY-SA impose), PII considerations, recommended and discouraged uses, and citations.

| # | Dataset (HF ID) | Role in `kbqa` | License |
|---|---|---|---|
| 1 | `rajpurkar/squad_v2` | Extractive reader supervision + abstention | CC-BY-SA-4.0 |
| 2 | `sentence-transformers/natural-questions` | Dense retriever (bi-encoder) pairs | CC-BY-SA-3.0 |
| 3 | `rag-datasets/rag-mini-wikipedia` | Self-contained demo KB + end-to-end QA | CC-BY-3.0 |
| 4 | `rag-datasets/rag-mini-bioasq` | Retrieval-recall benchmark (gold passage IDs) | CC-BY-2.5 |
| 5 | `neural-bridge/rag-dataset-12000` | Generative-RAG fine-tuning | Apache-2.0 |

No large data is committed to the repository. `scripts/download_data.py` pulls each set on demand via `datasets.load_dataset` into the HF cache; `data/.gitignore` blocks `*.parquet` / `*.arrow`. Only the small `rag-mini-wikipedia` demo index is materialized locally.

---

## 2. Intended Use

The system-level intended use is **grounded, cited extractive/abstractive question answering over an ingested document corpus, with safe abstention** when the corpus does not support an answer. The five datasets map onto that goal as follows:

- **`squad_v2`** — the reader backbone. Its ~50K **unanswerable** questions are the single most important asset for production-grade abstention: the extractive reader (`deepset/roberta-base-squad2`) learns to emit a null/`[CLS]` span rather than hallucinate. Used for fine-tuning and EM / F1 / NoAns-F1 evaluation.
- **`sentence-transformers/natural-questions`** (`pair` config) — clean `(query, positive passage)` pairs for the first bi-encoder fine-tune (`BAAI/bge-base-en-v1.5`) under Multiple-Negatives Ranking Loss; no joins required.
- **`rag-datasets/rag-mini-wikipedia`** — the default reproducible demo KB. The `text-corpus` config (3,200 passages) is indexed in FAISS; the `question-answer` config (918 QA pairs) drives end-to-end answer-quality evaluation. (QA rows carry **no gold passage IDs**, so it measures end-to-end answer quality only, not retrieval recall.)
- **`rag-datasets/rag-mini-bioasq`** — the retrieval-recall benchmark. Its `question-answer-passages` rows include `relevant_passage_ids`, enabling Recall@k / NDCG / MRR measurement against gold passages.
- **`neural-bridge/rag-dataset-12000`** — the cleanest-license (Apache-2.0) generative-RAG fine-tuning set for the optional `google/flan-t5-base` grounded reader (question + retrieved context → answer, with explicit "I don't know" supervision).

---

## 3. Composition

### 3.1 SQuAD v2 — `rajpurkar/squad_v2`

| Field | Value |
|---|---|
| Configs / splits (rows) | `squad_v2` / train **130,319** · validation **11,873** |
| Schema | `id`(str), `title`(str), `context`(str), `question`(str), `answers`={`text`:list[str], `answer_start`:list[int32]} |
| Unanswerable signal | empty `answers.text` / `answers.answer_start` lists ⇒ question is unanswerable |
| Domain | English Wikipedia paragraphs; crowd-written reading-comprehension questions |

~50K of the training questions are deliberately unanswerable given their context — the abstention supervision.

### 3.2 Natural-Questions pairs — `sentence-transformers/natural-questions`

| Field | Value |
|---|---|
| Config / split (rows) | `pair` / train **100,231** |
| Schema | `query` (real Google search question), `answer` (positive Wikipedia passage) |
| Domain | Real anonymized Google search queries paired with Wikipedia evidence passages |

A sentence-transformers-curated derivative of Google's Natural Questions, reshaped into clean asymmetric retrieval pairs (MNRL-ready, no document joins).

### 3.3 rag-mini-wikipedia — `rag-datasets/rag-mini-wikipedia`

| Field | Value |
|---|---|
| Configs (rows) | `text-corpus` / passages **3,200** · `question-answer` / test **918** |
| Schema | corpus: `passage` text + `id`; QA: `question`, `answer` (**no gold passage IDs**) |
| Domain | A compact Wikipedia subset; designed as a self-contained mini-RAG benchmark |

### 3.4 rag-mini-bioasq — `rag-datasets/rag-mini-bioasq`

| Field | Value |
|---|---|
| Configs (rows) | `text-corpus` / passages **40,200** · `question-answer-passages` / test **4,719** |
| Schema | corpus: `passage`, `id`; QA: `question`, `answer`, `relevant_passage_ids` (string e.g. `"[9797, 11906]"`, parse with `ast.literal_eval`) |
| Domain | Biomedical literature passages (derived from the BioASQ challenge) |

The gold `relevant_passage_ids` make this the dataset for exact retrieval-recall (Recall@k / MRR / NDCG).

### 3.5 neural-bridge/rag-dataset-12000 — `neural-bridge/rag-dataset-12000`

| Field | Value |
|---|---|
| Config (rows) | `default` / train **9,600** · test **2,400** (12,000 total) |
| Schema | `context` (retrieved passage), `question`, `answer` |
| Domain | General-domain context–question–answer triples for generative RAG fine-tuning |

---

## 4. Collection & Provenance

| Dataset | Source data | Annotation method |
|---|---|---|
| SQuAD v2 | English Wikipedia articles | Crowdworkers wrote answerable questions + spans, then adversarial unanswerable questions designed to look answerable |
| NQ pairs | Real Google search queries + Wikipedia | Derived from Google Natural Questions; re-curated by the sentence-transformers project into query/positive pairs |
| rag-mini-wikipedia | Wikipedia subset | Community-assembled mini benchmark for RAG evaluation |
| rag-mini-bioasq | Biomedical literature (BioASQ lineage) | Questions mapped to gold relevant passage IDs from the BioASQ challenge |
| rag-dataset-12000 | General web/text contexts | Constructed by Neural Bridge as context-grounded QA triples for RAG fine-tuning |

All five are redistributed through the Hugging Face Hub and pulled at runtime; `kbqa` performs no primary data collection.

---

## 5. Licensing & Attribution Obligations

| Dataset | License (SPDX) | Class | Obligation |
|---|---|---|---|
| `neural-bridge/rag-dataset-12000` | **Apache-2.0** | Permissive | Retain license + NOTICE; **cleanest for commercial use** |
| `rajpurkar/squad_v2` | **CC-BY-SA-4.0** | Attribution + ShareAlike | Attribute; derivatives/redistribution under same/compatible license |
| `sentence-transformers/natural-questions` | **CC-BY-SA-3.0** | Attribution + ShareAlike | Attribute; ShareAlike on redistribution |
| `rag-datasets/rag-mini-wikipedia` | **CC-BY-3.0** | Attribution | Attribute on use/redistribution (no ShareAlike) |
| `rag-datasets/rag-mini-bioasq` | **CC-BY-2.5** | Attribution | Attribute on use/redistribution (no ShareAlike) |

**Attribution requirements (CC-BY / CC-BY-SA).** Wherever `kbqa` is distributed, the product documentation and a `NOTICE`/`DATA_LICENSES` file MUST: (1) credit each dataset's creators, (2) link to the source and to the license deed, and (3) indicate if the data were modified (e.g. SQuAD contexts deduped into a KB, NQ reshaped into pairs).

**ShareAlike (CC-BY-SA-4.0, CC-BY-SA-3.0 → SQuAD v2 and NQ pairs).** If any redistributed *dataset-like* artifact is a derivative of these (e.g. a published SQuAD-derived passage pool, or a public dump of mined pairs), that artifact must be released under the same or a compatible CC-BY-SA license. Trained **model weights** are generally treated as a separate, non-dataset artifact, but **do not redistribute derived corpora** built from CC-BY-SA sources under a more restrictive license. The Apache-2.0 set (`rag-dataset-12000`) carries no ShareAlike constraint and is preferred when a clean redistribution license is needed.

> Note on the broader project: other candidate datasets considered in the design (e.g. `mandarjoshi/trivia_qa`, the MS MARCO family) carry **unknown / research-only** licenses and are **out of scope for this card** and flagged for legal sign-off before any commercial use. The five datasets documented here are the ones `kbqa` actually ships against.

---

## 6. PII & Privacy Considerations

- **No deliberate personal data.** All five datasets are built from public Wikipedia, public biomedical literature, or synthetically/curated QA triples. None is a dataset *about* individuals.
- **Incidental PII.** Wikipedia- and web-derived text inevitably mentions public figures and incidental named entities. SQuAD contexts, NQ passages, and the rag-mini corpora may contain names, dates, and biographical facts about public persons. These are public-record mentions, not sensitive private data.
- **Search-query provenance (NQ).** `sentence-transformers/natural-questions` derives from real Google search queries; upstream Natural Questions queries were **anonymized** before release. Treat query strings as already de-identified but avoid attempting re-identification.
- **Biomedical text (bioasq).** Passages are drawn from published biomedical literature (abstracts/articles), **not** patient records; no PHI is expected. Do not treat any content as clinical advice.
- **Mitigation in `kbqa`.** The system answers only from the *ingested* corpus and cites source spans; it does not exfiltrate training data, and the faithfulness gate prevents emission of unsupported (potentially fabricated personal) claims.

---

## 7. Recommended Uses

- Fine-tuning and evaluating extractive reading comprehension **with abstention** (SQuAD v2).
- Fine-tuning dense retrievers for asymmetric QA retrieval (NQ pairs).
- Building a small, fully reproducible demo knowledge base and end-to-end QA evaluation (rag-mini-wikipedia).
- Measuring retrieval Recall@k / NDCG@10 / MRR@10 against gold passages (rag-mini-bioasq).
- Fine-tuning a grounded generative reader that says "I don't know" when context is insufficient (rag-dataset-12000).
- Academic / research reporting, with proper attribution.

## 8. Discouraged / Out-of-Scope Uses

- **Clinical or medical decision-making** from rag-mini-bioasq — it is a retrieval benchmark, not validated medical knowledge.
- **Treating answers as ground truth** without the citation + faithfulness checks `kbqa` enforces.
- **Re-identification** of individuals from NQ queries or Wikipedia mentions.
- **Redistributing CC-BY-SA-derived corpora under incompatible/more-restrictive licenses**, or stripping attribution.
- **Assuming retrieval recall** can be measured on rag-mini-wikipedia QA — those rows have **no gold passage IDs**; use rag-mini-bioasq (or SQuAD-derived gold contexts) for recall.
- Using these datasets to train systems that generate medical, legal, or financial advice for end users without domain validation.

---

## 9. Maintenance & Versioning

- Datasets are pinned by HF ID + config + split in `scripts/download_data.py`; row counts above were verified 2026-06-26 and should be re-checked if upstream revisions change.
- The `kbqa` FAISS index records `model_version` in its `manifest.json`; data and embedding model are versioned together and re-indexed on any encoder change.
- A `data/DATA_LICENSES` file in the repo enumerates the per-dataset attribution strings required by §5.

---

## 10. Citations (BibTeX)

```bibtex
@inproceedings{rajpurkar2018squad2,
  title     = {Know What You Don't Know: Unanswerable Questions for {SQuAD}},
  author    = {Rajpurkar, Pranav and Jia, Robin and Liang, Percy},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association
               for Computational Linguistics (ACL)},
  pages     = {784--789},
  year      = {2018}
}

@article{kwiatkowski2019natural,
  title   = {Natural Questions: A Benchmark for Question Answering Research},
  author  = {Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia
             and Collins, Michael and Parikh, Ankur and Alberti, Chris and
             Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and
             Lee, Kenton and Toutanova, Kristina and Jones, Llion and
             Kelcey, Matthew and Chang, Ming-Wei and Dai, Andrew M. and
             Uszkoreit, Jakob and Le, Quoc and Petrov, Slav},
  journal = {Transactions of the Association for Computational Linguistics (TACL)},
  volume  = {7},
  pages   = {453--466},
  year    = {2019}
}

@misc{ragmini_wikipedia,
  title        = {rag-mini-wikipedia},
  author       = {{RAG Datasets}},
  howpublished = {Hugging Face Hub, rag-datasets/rag-mini-wikipedia},
  note         = {Licensed CC-BY-3.0},
  year         = {2024}
}

@inproceedings{tsatsaronis2015bioasq,
  title     = {An overview of the {BIOASQ} large-scale biomedical semantic
               indexing and question answering competition},
  author    = {Tsatsaronis, George and Balikas, Georgios and Malakasiotis,
               Prodromos and others},
  booktitle = {BMC Bioinformatics},
  volume    = {16},
  number    = {1},
  pages     = {138},
  year      = {2015},
  note      = {Mini subset: Hugging Face Hub, rag-datasets/rag-mini-bioasq, CC-BY-2.5}
}

@misc{neuralbridge2024ragdataset12000,
  title        = {RAG Dataset 12000},
  author       = {{Neural Bridge AI}},
  howpublished = {Hugging Face Hub, neural-bridge/rag-dataset-12000},
  note         = {Licensed Apache-2.0},
  year         = {2024}
}
```

---

*This card documents data governance for the `kbqa` system. All IDs, configs, splits, and licenses are VERIFIED against the Hugging Face Hub (2026-06-26). Anything not verifiable upstream must not be asserted downstream.*
