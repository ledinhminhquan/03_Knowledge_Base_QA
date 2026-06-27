# Ethics & Responsible AI

*Project #3 — Knowledge Base Question-Answering System (`kbqa`). Author: Le Dinh Minh Quan (23127460). Assignment §11.*

This system answers natural-language questions over an organization's documents using grounded, cited Retrieval-Augmented Generation (RAG). Because it sits between people and the information they act on, its design decisions are also ethical decisions. This statement records who benefits, who could be harmed, the bias and fairness risks we anticipate, how non-technical stakeholders can understand each answer, and the safeguards against misuse. The recurring theme is that **the system is built to abstain rather than guess**: a grounded reader, a faithfulness entailment gate, and a "I don't know" path are the load-bearing ethical controls, not afterthoughts.

## 1. Who Benefits

| Beneficiary | Benefit |
|---|---|
| Employees / internal users | Fast, sourced answers over large internal corpora (PDFs, wikis, tickets) instead of manual search. Latency targets are p50 ~350 ms / p95 ~800 ms on CPU for the extractive reader (no GPU, no paid API), so the tool is cheap to run and broadly accessible. |
| End users / customers | Answers arrive **with citations and a confidence score**, so a person can verify a claim against the original passage rather than trusting a black box. |
| The organization | Zero paid-API, CPU-default deployment lowers cost and keeps data on-premise (FAISS index + documents stay local), reducing exposure compared to sending corpora to a third-party LLM. |
| Knowledge owners | Idempotent ingest (SHA-256 dedup) and `model_version`-pinned indexes make provenance and reproducibility auditable. |

The benefit is concrete: every `/ask` response carries `answer`, `citations[{chunk_id, source, quote, offset}]`, and `confidence`. The value is not just the answer but the **traceable path back to the source**.

## 2. Who Could Be Harmed

- **Users misled by wrong-but-confident answers.** The single largest risk in any RAG system is a fluent, authoritative-sounding answer that is unsupported. We mitigate this directly: the generative reader is instructed to answer using *only* retrieved context, the extractive reader (`deepset/roberta-base-squad2`) uses a **null-score abstention** mechanism, and a downstream **faithfulness/entailment gate** must confirm the answer is supported by cited passages before it is emitted. If not, the system returns *"I don't have enough information in the knowledge base."* Thresholds `TAU_HIGH=0.55` and `TAU_LOW=0.15` route low-support cases toward abstention rather than fabrication.
- **Over-reliance and automation bias.** Users may stop verifying because the tool is usually right. Citations and confidence are deliberately surfaced *with* every answer to keep a human in the loop; the UI (Gradio/`/ask`) shows sources and confidence beside the answer, nudging verification rather than blind trust. For high-stakes use we recommend a human-in-the-loop policy (see §3).
- **People excluded by an English-only system.** The retriever, reranker, and readers (`bge-base-en-v1.5`, `ms-marco-MiniLM-L-6-v2`, `roberta-base-squad2`, `flan-t5-base`) are English-centric. Non-English speakers and non-English documents are effectively unsupported, which can disadvantage parts of a workforce or user base. This is a known limitation; multilingual encoders (e.g. `bge-reranker-v2-m3` is multilingual) are a documented future extension, not a current guarantee.
- **People affected by stale or partial corpora.** An answer is only as current as the indexed documents. The system can answer confidently from outdated passages. Mitigation: citations expose the source (and its metadata `ingested_at`), so a reader can judge recency; abstention triggers when the corpus lacks support.

## 3. Bias & Fairness

RAG inherits the biases of its corpus and its retrieval distribution:

- **Corpus bias.** If the indexed documents over-represent certain viewpoints, products, regions, or authors, answers will reflect that skew. The system does not correct corpus bias; it propagates it. The honest control is transparency — citations let a reader see *which* source produced a claim and weigh it accordingly.
- **Retrieval bias toward majority topics.** Dense retrieval favors well-covered, high-frequency topics; rare entities and minority topics retrieve worse. We deliberately fuse **BM25 (sparse) with dense `bge` retrieval via Reciprocal Rank Fusion (RRF)** precisely because BM25 catches exact-term and rare-entity matches that dense encoders miss, partially counteracting majority-topic bias. A cross-encoder reranker then re-scores the top candidates.

**Mitigations (and their limits):**

1. **Citations enable verification** — bias is exposed rather than hidden, so a human can challenge it.
2. **Abstention** — when support is weak (common for under-represented topics), the system says "I don't know" instead of producing a confidently biased guess.
3. **Human-in-the-loop for high-stakes use** — for medical, legal, financial, or HR questions, the system is positioned as a *retrieval aid that surfaces sourced passages*, not an autonomous decision-maker. A qualified human must make the final call.

These mitigations reduce harm; they do not eliminate bias. A biased corpus produces biased-but-cited answers, and the responsibility for corpus curation and fairness auditing remains with the deploying organization.

## 4. Explainability for Non-Technical Stakeholders

The core explainability promise is simple enough for any stakeholder: **every answer shows its sources and a confidence score.**

```
Question ──> [retrieve + rerank] ──> [grounded reader] ──> [faithfulness gate]
                                                                  │
                          ┌───────────────────────────────────────┴───────────┐
                          ▼                                                     ▼
            Answer + Citations + Confidence                      "I don't know" (abstain)
            (each citation = source + quoted span + offset)
```

A non-technical user does not need to understand embeddings or cross-encoders. They see: the answer, the exact passages it came from (with a quoted span and source document), and a confidence number. They can click through to verify. When the system is unsure, it abstains visibly rather than failing silently. Optionally, a decision `trace` (`return_trace`) records every step — analyze, retrieve, rerank, sufficiency, generate, faithfulness — for auditors who *do* want the internals. This makes the system's reasoning inspectable at two levels: a citation-and-confidence view for everyone, and a full trace for reviewers.

## 5. Potential Misuse & Safeguards

| Misuse | Safeguard |
|---|---|
| **Surveillance over private documents** — indexing employees' private files, messages, or HR records to answer questions about individuals. | Access control is the deployer's responsibility: ingest only authorized corpora; apply document-level `filters` and per-source permissions; do not index personal data without consent and lawful basis. The system is built for organizational knowledge, not people-profiling. |
| **Generating authoritative-looking misinformation** — using the cited, confident format to lend false credibility to a manipulated corpus. | Citations point to *actual indexed passages*; the faithfulness gate blocks claims not entailed by those passages. An answer can only be as trustworthy as its visible sources, which a reader can inspect. Corpus integrity (signed, curated ingest) is the deployer's safeguard. |
| **Scraping / bulk extraction** — using `/ask` or `/batch` to exfiltrate a protected corpus. | Rate-limiting, authentication, and `max_concurrency` caps on `/batch`; index access gated behind the API; respect source licenses (the data stack flags CC-BY-SA attribution/share-alike obligations and license-unknown sets for legal sign-off before commercial use). |
| **Removing the abstention path** to force an answer for every query. | Abstention is architectural (null-score reader + entailment gate + `TAU` thresholds), not cosmetic. Disabling it is a deliberate, documented downgrade and should require explicit sign-off, because it directly increases hallucination risk. |
| **Over-automation of high-stakes decisions.** | Documented human-in-the-loop requirement for high-stakes domains; confidence and citations are mandatory in the response contract to keep a person able to verify. |

## 6. Summary

The ethical posture of `kbqa` is **grounded, cited, and abstaining by default**. Benefits — fast, sourced, low-cost, on-premise answers — are real and broad. The principal harms (confident errors, automation bias, English-only exclusion, corpus and retrieval bias, and misuse for surveillance or misinformation) are met with concrete, architectural controls: an entailment-gated grounded reader, mandatory citations and confidence, hybrid retrieval to counter majority-topic skew, and an explicit "I don't know" path. None of these controls removes the deployer's responsibility for corpus curation, access control, license compliance, and human oversight in high-stakes settings. Responsible operation is a shared contract between the system's design and the organization that runs it.
