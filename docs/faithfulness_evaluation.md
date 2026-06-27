# Faithfulness & Groundedness Evaluation

**Project:** Knowledge Base Question-Answering System (`kbqa`) — RAG-over-documents
**Author:** Le Dinh Minh Quan (23127460)
**Scope:** This document is the RAG analogue of a quality/fairness audit. Where a classifier audit measures bias and error parity across groups, a RAG audit measures whether generated answers are *actually supported by the cited evidence* — i.e., whether the system is honest about what the knowledge base does and does not contain. It defines the metrics, specifies how the `kbqa.agent.tools.check_faithfulness` module computes them, describes citation-accuracy and hallucination/abstention error analysis, gives an illustrative (clearly-marked **projected**) results table, and shows how the verified thresholds drive the abstain decision.

---

## 1. Why faithfulness is the central safety metric

A RAG system has two ways to be wrong:

1. **Retrieval failure** — the right passage was never surfaced (covered by Recall@k / NDCG / MRR).
2. **Generation failure** — the right passage *was* surfaced, but the answer says something the passage does not support (a **hallucination**).

Answer-quality metrics like EM/F1 conflate both and reward fluent guesses. A model can produce a fluent, plausible, *wrong* answer that scores partial F1. For a production assistant the more dangerous failure is the confident hallucination, because users trust cited answers. Faithfulness evaluation isolates generation failure and is the gate that decides between **emit a cited answer** and **abstain ("I don't know")**.

The design contract for `kbqa`: the system **never answers from parametric memory**. Every emitted answer must be entailed by the retrieved context that is cited alongside it. If entailment cannot be established, the system abstains.

---

## 2. Definitions

We adopt the now-standard RAG-triad decomposition (the three vertices are *query*, *context*, *answer*) plus a citation-level metric.

| Metric | Question it answers | Formal definition (per query) |
|---|---|---|
| **Faithfulness / Groundedness** | Is the answer entailed by the cited context? | Fraction of answer claims (sentences/atomic facts) that are *entailed* by the union of cited passages. `faithfulness = supported_claims / total_claims`. |
| **Answer relevance** | Does the answer actually address the question? | Semantic alignment between the question and the answer (penalizes evasive, partial, or off-topic answers even when grounded). |
| **Context precision** | Of the retrieved/cited passages, how many are relevant? | `relevant_retrieved / total_retrieved` over the reranked top-n that the reader saw. High precision ⇒ the reader is not being distracted by noise. |
| **Context recall** | Of the evidence needed, how much was retrieved? | `relevant_retrieved / total_relevant` against gold supporting passages. Bounds the best achievable faithfulness — you cannot ground a claim whose evidence was never retrieved. |
| **Citation accuracy** | Are the cited `chunk_id`s the *right* passages? | Precision/recall of emitted citation markers vs. gold supporting chunks (Section 5). |

**Key relationships.**
- Faithfulness is *conditioned on the cited context*: an answer can be perfectly faithful to a wrong passage. That is why it is paired with **context recall/precision** (did we retrieve the right thing?) and **answer relevance** (did we answer the actual question?). All three are needed; none alone is sufficient.
- Context **recall** is the ceiling on end-to-end correctness; context **precision** is the ceiling on reader robustness. For `kbqa` the reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, top-50 → top-5) is the primary precision lever, and hybrid BM25 ⊕ bge dense retrieval fused by RRF is the primary recall lever.

---

## 3. How the faithfulness module measures groundedness

The agent's faithfulness gate is `check_faithfulness(answer, ctx) -> {supported, support_score}` (decision point 3 in the state machine; Self-RAG `ISSUP` reflection). It treats the concatenated **cited** chunks as the *premise* and each answer claim as the *hypothesis*, and asks: does the premise entail the hypothesis?

### 3.1 Primary path: NLI entailment

The intended primary scorer is a natural-language-inference cross-encoder that outputs `P(entail | premise, hypothesis)`. An answer claim is **supported** when `P(entail) >= TAU_NLI`. The per-answer `support_score` is the mean (or min, for strict mode) over claims.

> **IMPORTANT — unverified model id.** The NLI model id contemplated in the design (`cross-encoder/nli-deberta-v3-small`) was **not** verified against the Hugging Face Hub during research and **must not be asserted as existing**. It is wired behind a config flag and verified-on-load; if the id does not resolve, the module automatically falls back to the path in §3.2. Do not ship the NLI path until the id is confirmed live on the Hub.

### 3.2 Verified fallback path (default in this build)

Because the NLI id is unverified, the **default, always-available** groundedness scorer is built entirely from models verified live on the Hub (2026-06-26). It blends two signals into `support_score`:

1. **Embedding-overlap entailment proxy** — using the verified retriever `BAAI/bge-base-en-v1.5` (MIT, 768-dim), embed each answer claim and each cited-passage sentence; a claim is grounded if its maximum cosine similarity to any cited sentence exceeds a tuned threshold. This is a soft, paraphrase-tolerant proxy for entailment.
2. **Reranker groundedness proxy** — re-score the pair *(answer claim, cited passage)* with the verified `cross-encoder/ms-marco-MiniLM-L-6-v2`. A high relevance score is a strong correlate of "this passage is about this claim."
3. **Lexical overlap** — content-word (lemma) overlap and number/entity match as a cheap, interpretable backstop that catches the common failure mode of a *number or named entity in the answer that appears nowhere in the cited text* (e.g., a fabricated date).

`support_score` is a calibrated blend of (1), (2), and (3); `supported = support_score >= TAU_SUPPORT`. All three components run on CPU, consistent with the CPU-default constraint.

```
                cited chunks (premise)        answer claim (hypothesis)
                        │                              │
         ┌──────────────┴───────────────┐              │
         ▼                              ▼              ▼
   bge-base-en-v1.5            ms-marco-MiniLM-L-6-v2  lexical/entity
   embedding cosine            reranker pair score     overlap
         └──────────────┬───────────────┴──────────────┘
                        ▼
                 calibrated support_score  ──►  supported = score ≥ TAU_SUPPORT
                        │
              (optional, if id verified)
                        ▼
            NLI P(entail) ≥ TAU_NLI  [config-gated]
```

### 3.3 Claim decomposition

For extractive answers (`deepset/roberta-base-squad2`) the "claim" is a single span, so faithfulness reduces to: *is the span a substring of (or entailed by) a cited chunk?* — near-trivially groundable, which is the safety advantage of the extractive default. For generative answers (`google/flan-t5-base`, grounded + IDK) the answer may contain multiple sentences; it is split into atomic claims and each is scored independently, so a single unsupported sentence lowers the score (strict mode can reject the whole answer).

---

## 4. The faithfulness gate and the abstain decision

Faithfulness is not only measured offline; it is an **online gate** in the agent state machine. Two verified thresholds govern the loop (`TAU_HIGH = 0.55`, `TAU_LOW = 0.15`), applied to the top **rerank** score in the sufficiency check, and a faithfulness threshold gates emission.

| Stage | Signal | Threshold | Action |
|---|---|---|---|
| Sufficiency (decision pt 2) | top rerank score | `>= TAU_HIGH (0.55)` | **SUFFICIENT** → proceed to generate |
| Sufficiency | top rerank score | `[TAU_LOW, TAU_HIGH)` i.e. `[0.15, 0.55)` | **AMBIGUOUS** → rewrite + expand query, widen `top_k`, retry (CRAG loop, bounded by `max_iterations = 3`) |
| Sufficiency | top rerank score | `< TAU_LOW (0.15)` | **INSUFFICIENT** → decompose further, or after `max_iterations` ask a clarifying question / abstain |
| Faithfulness (decision pt 3) | `support_score` | `< TAU_SUPPORT` | **ABSTAIN** — drop the unsupported claim; if no supported claims remain, emit "I don't know" |
| Reader (extractive) | null score vs. best span | `null - best > threshold` | **ABSTAIN** — `roberta-base-squad2` native no-answer |

So there are **three independent ways to abstain**, defense-in-depth:

1. **Routing** — `analyze_query` classifies the query as `unanswerable` ⇒ immediate "I don't know".
2. **Retrieval/sufficiency** — top rerank score never clears `TAU_HIGH` even after the CRAG correction loop exhausts `max_iterations`.
3. **Faithfulness** — context cleared the bar but the generated answer is not entailed by it ⇒ the gate vetoes emission.

For multi-hop questions a **final global faithfulness gate** runs on the synthesized answer after the per-sub-question gates, and `confidence = min` over hops. Any required hop that stays INSUFFICIENT after `max_iterations` forces an abstain (optionally surfacing the supported partial answer). The system **never fabricates** a missing fact (e.g., a founding year) from parametric memory — the gate requires entailment from retrieved context.

---

## 5. Citation accuracy

A grounded answer is only auditable if its citations point at the *right* passages. The `/ask` response attaches `citations[{marker, chunk_id, doc_id, source, quote, offset}]`; we evaluate these markers against gold supporting chunks.

- **Citation precision** = (cited chunks that are gold-relevant) / (all cited chunks). Penalizes "citation padding."
- **Citation recall** = (gold-relevant chunks that were cited) / (all gold-relevant chunks). Penalizes claims supported by uncited evidence.
- **Citation-span validity** = fraction of emitted `quote`/`span` fields that are genuine substrings of (or entailed by) the referenced `chunk_id` text. Catches the "right chunk, wrong quote" failure.

Gold supporting chunks come from datasets that carry passage ids — `rag-datasets/rag-mini-bioasq` (`relevant_passage_ids`, parsed with `ast.literal_eval`) — and from the SQuAD-derived KB fallback, where each QA item has a known gold context, giving exact citation ground truth at zero extra download. `rag-mini-wikipedia` QA carries **no** gold passage ids, so on that corpus we report end-to-end answer faithfulness only, not citation recall.

---

## 6. Hallucination & abstention error analysis

We classify every evaluated answer into a confusion-style grid driven by two axes: *did the system answer or abstain?* and *was answering the correct choice?* (i.e., is the question answerable from the KB, and was the answer faithful?).

| | Should answer (answerable) | Should abstain (unanswerable) |
|---|---|---|
| **System answered, faithful** | ✅ Correct grounded answer | n/a (no gold support to be faithful to) |
| **System answered, unfaithful** | ❌ **Hallucination** (the critical error) | ❌ **Hallucination** (worst case: confident answer to an unanswerable question) |
| **System abstained** | ⚠️ **Over-abstention** (missed answerable) | ✅ Correct abstention |

**Error types we track and triage:**

- **Hallucination rate** — fraction of *answered* queries whose answer is not faithful (`supported = false`). Target: drive toward zero, accepting higher abstention as the trade. This is the headline safety number.
- **Over-abstention rate** — fraction of *answerable* queries the system declined. Pure-abstention systems are trivially non-hallucinating but useless; this metric keeps the abstain thresholds honest.
- **Abstain rate** (overall) and **NoAns-F1** — measured on SQuAD v2 unanswerables (the ~50K no-answer items are exactly why SQuAD v2 is the reader backbone) and reported separately from HasAns-F1.
- **Citation errors** — right answer but wrong/missing citation; surfaces silently-correct-but-unauditable answers.

Every decision (route, each sufficiency verdict, each faithfulness verdict, abstain reason) is appended to `state.trace`, so each error case is reproducible from the audit log rather than guessed at.

---

## 7. Illustrative results (PROJECTED — not measured)

> ⚠️ **The table below is illustrative and clearly marked PROJECTED.** These are *target* operating points used to design thresholds and the report narrative, **not** measured results. Real numbers are produced by the evaluation harness in Section 8 and must replace these before any external claim. Datasets/models are the verified ids from the design brief; the numbers are not.

**Faithfulness & groundedness, projected operating points (demo KB `rag-mini-wikipedia` + SQuAD-derived gold contexts):**

| Configuration | Faithfulness | Answer relevance | Ctx precision | Ctx recall | Citation P / R | Hallucination rate | Abstain rate |
|---|---|---|---|---|---|---|---|
| BM25 + zero-shot reader (floor) | 0.71 | 0.74 | 0.52 | 0.69 | 0.61 / 0.58 | 0.18 | 0.07 |
| + bge dense + RRF hybrid | 0.79 | 0.78 | 0.63 | 0.81 | 0.70 / 0.69 | 0.12 | 0.09 |
| + cross-encoder rerank (top-50→5) | 0.86 | 0.81 | 0.78 | 0.83 | 0.79 / 0.76 | 0.08 | 0.10 |
| + faithfulness gate (full stack) | **0.94** | 0.83 | 0.79 | 0.84 | 0.84 / 0.80 | **0.03** | 0.14 |

**How to read it (intended trend, to be validated):** each row adds one component. The reranker is the largest single jump in context precision (0.63 → 0.78), which in turn lifts faithfulness. The faithfulness gate trades a higher abstain rate (0.10 → 0.14) for a much lower hallucination rate (0.08 → 0.03) — exactly the intended safety/coverage trade. The job of the real evaluation is to confirm the *direction* of these arrows and find the real magnitudes.

---

## 8. Evaluation harness & protocol

| Dimension | Metric | Instrument / data |
|---|---|---|
| Faithfulness / groundedness | % answer claims entailed by cited passages | §3 scorer (NLI if verified, else bge+reranker+lexical fallback) over (claim, cited chunk) |
| Answer relevance | question↔answer semantic alignment | `bge-base-en-v1.5` cosine; penalize evasive answers |
| Context precision / recall | relevant-retrieved ratios | gold passage ids: `rag-mini-bioasq` `relevant_passage_ids`; SQuAD-derived gold contexts |
| Citation accuracy | precision / recall / span validity | marker→`chunk_id` set overlap vs. gold (§5) |
| Hallucination / abstention | hallucination rate, over-abstention, abstain rate, NoAns-F1 | confusion grid (§6); SQuAD v2 unanswerables |
| Latency (context) | p50/p95 `/ask` | `/metrics` histograms; extractive ~350/800 ms CPU |

**Protocol.** (1) Establish the BM25 + zero-shot floor. (2) Add components one at a time (hybrid → rerank → gate), re-measuring every faithfulness metric so each component's contribution is attributable. (3) Run the abstention battery on SQuAD v2 unanswerables and report HasAns-F1 / NoAns-F1 separately. (4) Sweep `TAU_SUPPORT` (and `TAU_NLI` when the NLI id is verified) on a held-out dev set to set the hallucination/over-abstention operating point; `TAU_HIGH`/`TAU_LOW` for the sufficiency loop stay pinned at the verified `0.55` / `0.15`. (5) Replace the §7 projected table with measured numbers before any external report claim.

---

## 9. Threats to validity

- **Faithfulness ≠ correctness.** An answer faithful to a *wrong* retrieved passage still passes the gate. This is bounded by context recall/precision, which we report alongside — never faithfulness in isolation.
- **Proxy entailment is soft.** The verified fallback (embedding + reranker + lexical) approximates entailment and can be fooled by high lexical overlap with the *wrong* relation. The lexical/entity-match component specifically guards the fabricated-number/-entity case; the NLI path (once its id is verified) is the rigorous upgrade.
- **No gold citations on `rag-mini-wikipedia`.** Citation recall there is unmeasurable; we lean on `rag-mini-bioasq` and SQuAD-derived gold for citation ground truth.
- **Threshold transfer.** Operating points tuned on the demo KB may not transfer to a customer corpus; thresholds are config, re-tunable per deployment, and `MODEL_VERSION` pins the encoder/reranker/reader/index together so a re-tune is never silently mixed across index versions.
