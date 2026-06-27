# Knowledge Base Question-Answering System (RAG)

A **production, agentic Retrieval-Augmented Generation** system that answers
natural-language questions over a document knowledge base with **grounded, cited
answers — and safely abstains** ("I don't know") when the corpus lacks support.
Built for the *NLP in Industry* final assignment (Project #3).

> **Pipeline:** `ingest → chunk → index (FAISS) → analyze query → retrieve
> (BM25 ⊕ dense, RRF) → rerank → sufficiency check → grounded generate w/ citations
> → faithfulness gate → answer | abstain`

It runs **fully on CPU with zero paid APIs** (BM25 + a small bi-encoder + an
extractive reader + a deterministic agent) and transparently upgrades to a
fine-tuned retriever/reader, a cross-encoder reranker and an optional LLM brain.

> **Why RAG-over-documents (not ChatKBQA semantic parsing)?** The reference
> ChatKBQA parses questions to SPARQL over Freebase — accurate but impractical to
> deploy (needs a ~50 GB Freebase triplestore + per-schema fine-tuning). We build
> RAG over documents (drop in PDFs/wikis → re-embed, citations, safe abstention)
> and keep semantic parsing as a **pluggable `kg_query` backend**. See
> [docs/agent_architecture.md](docs/agent_architecture.md).

---

## ✨ Highlights

| Requirement (assignment) | How this project delivers it |
|---|---|
| Trainable model + baseline | Fine-tuned **bi-encoder retriever** + **extractive reader** vs **BM25** baseline |
| Hyperparameter tuning | retriever hard-negative mining + reader **null-score threshold** sweep |
| Deployment | **FastAPI** (`/ingest /search /ask /batch`) + **Gradio** demo + Docker/HF Space |
| Agentic AI component | CRAG correction loop + Self-RAG reflection, 3 decision points, full audit trace |
| Continual learning & monitoring | query logs + drift (PSI) + re-index / re-mine strategy |
| Data privacy & robustness | PII redaction at ingest, **prompt-injection** defense, abstention |
| Ethics & faithfulness | citations for verification, groundedness gate, human-in-the-loop |
| Reproducible repo | `src/ data/ models/ configs/ tests/ docs/`, Docker, CI |
| Auto report + slides | one-button **autopilot** → `report.pdf` + `slides.pptx` |

---

## 🗂️ Repository structure

```
03_Knowledge_Base_QA/
├── src/kbqa/
│   ├── config.py            # typed config + YAML loader
│   ├── cli.py               # `kbqa` entrypoint (all commands)
│   ├── data/                # datasets, chunking, corpus/KB builder, samples
│   ├── index/               # FAISS vector store (build / persist / load)
│   ├── models/              # retriever, reranker, extractive + generative readers, BM25
│   ├── training/            # train retriever/reader/generator, tune, evaluate
│   ├── agent/               # state, tools, policy, orchestrators, RAG agent (CRAG loop)
│   ├── api/                 # FastAPI app, schemas, Gradio UI, combined app
│   ├── analysis/            # error analysis, faithfulness, latency
│   ├── autoreport/          # charts + PDF report + PPTX slides
│   ├── monitoring/          # query-log aggregation + drift report
│   ├── automation/          # one-button autopilot
│   └── grading/             # rubric completeness self-check
├── configs/ · data/ · models/ · tests/ · docs/ · notebooks/ · app/ · deploy/ · scripts/ · sample_data/
├── Dockerfile · docker-compose.yml · Makefile
├── pyproject.toml · requirements.txt · requirements_colab.txt · README.md
```

---

## 📚 Data & models (verified, public)

| Stage | Dataset (HF) | License | Model (HF) | License |
|---|---|---|---|---|
| Reader (extractive + abstain) | [`rajpurkar/squad_v2`](https://huggingface.co/datasets/rajpurkar/squad_v2) | CC-BY-SA-4.0 | `deepset/roberta-base-squad2` | CC-BY-4.0 |
| Retriever pairs | [`sentence-transformers/natural-questions`](https://huggingface.co/datasets/sentence-transformers/natural-questions) | CC-BY-SA-3.0 | bi-encoder `BAAI/bge-base-en-v1.5` | MIT |
| Reranking | — | — | cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 |
| Demo KB | [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) | CC-BY-3.0 | generative `google/flan-t5-base` | Apache-2.0 |
| Sparse baseline | — | — | **BM25** (`rank_bm25`) | — |

No large data is committed; download with `kbqa data --task all`. Details:
[docs/data_description.md](docs/data_description.md).

---

## 🚀 Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

# Try the agent immediately on the built-in sample KB (no download needed)
kbqa demo-agent --config configs/infer.yaml
kbqa ask --question "Who designed the Eiffel Tower?" --config configs/infer.yaml

# Launch the demo UI (http://localhost:7860)
python app/gradio_app.py
```

### Build the KB, train, evaluate

```bash
kbqa data --task all                                  # download + build demo KB index
kbqa build-kb        --config configs/train.yaml
kbqa train-retriever --config configs/train.yaml      # fine-tune bi-encoder (MNRL + hard negatives)
kbqa train-reader    --config configs/train.yaml      # fine-tune extractive reader (SQuAD v2)
kbqa evaluate        --config configs/train.yaml      # retrieval recall + answer EM/F1 + abstention
```

### Serve the REST API

```bash
kbqa serve --config configs/infer.yaml --host 0.0.0.0 --port 8000   # http://localhost:8000/docs
```

```bash
# add documents to the KB
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"documents":[{"title":"Paris","text":"Paris is the capital of France."}]}'

# ask a grounded, cited question
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What is the capital of France?"}'
```

---

## 🖥️ Train on Google Colab (H100 / flexible GPU)

Open [`notebooks/KBQA_Colab_Training_H100_AUTOPILOT.ipynb`](notebooks/KBQA_Colab_Training_H100_AUTOPILOT.ipynb).
It auto-detects the GPU (H100/A100/L4/T4), redirects HF caches + checkpoints to
Drive (survives disconnects, **resumes**), installs Colab-safe deps, fine-tunes
the retriever + reader, evaluates, and **auto-generates the report + slides**.
Step-by-step Drive layout and testing instructions:
[`notebooks/COLAB_GUIDE.md`](notebooks/COLAB_GUIDE.md).

---

## 🤖 Agentic component

A deterministic state machine combining **Rewrite-Retrieve-Read**, **Corrective-RAG**
(sufficiency correction loop) and **Self-RAG** (reflection). Three decision points
act on the model's own intermediate outputs:

1. **Analyze** — route simple / multi-hop / unanswerable (+ query rewrite/decompose).
2. **Sufficiency** — SUFFICIENT / AMBIGUOUS (widen + retry) / INSUFFICIENT (abstain), bounded by `max_iterations`.
3. **Faithfulness** — emit a cited answer only if it's grounded in the context, else **abstain**.

Switch brains with `agent.orchestrator: rule | llm`. The optional LLM brain
validates its output and **falls back to rules** on any error. Worked multi-hop
example + diagram: [docs/agent_architecture.md](docs/agent_architecture.md).

---

## 🧰 One-button autopilot

```bash
kbqa autopilot --config configs/train.yaml \
  --title "Knowledge Base Question-Answering System" --author "Le Dinh Minh Quan"
```

Runs build-KB → train → evaluate → benchmark → error analysis → faithfulness →
**`report.pdf` + `slides.pptx`** + `grading_checklist.json` + `submission_bundle.zip`
under `artifacts/submission/submission-<timestamp>/`.

---

## 🧪 Tests

```bash
pytest -q          # CPU-only; runs the BM25 fallback path (no GPU / model download)
```

---

## 📖 Documentation (assignment deliverables)

Problem · Data · Model Selection · Deployment · **Agent Architecture** · Continual
Learning & Monitoring · Privacy & Robustness · Project Plan · Ethics · **Faithfulness
Evaluation** · System Architecture · Model/Data Card · Slide outline — all in
[`docs/`](docs/).

---

## 🚢 Deployment

* **Docker:** `docker compose up --build` → API + UI on port 7860.
* **Hugging Face Space (Docker SDK):** [deploy/README_HF_SPACE.md](deploy/README_HF_SPACE.md).
* **Endpoints:** `/health`, `/ingest`, `/search`, `/ask`, `/batch`, `/metrics`.

## ⚖️ Responsible use

Answers are **grounded and cited**; the agent **abstains rather than hallucinate**,
redacts PII at ingest, and defends against prompt-injection in retrieved documents.
See [docs/ethics_statement.md](docs/ethics_statement.md) and
[docs/privacy_robustness.md](docs/privacy_robustness.md).

## License

MIT (project code) — see [LICENSE](LICENSE). Datasets/models keep their own licenses
(e.g. `roberta-base-squad2` is CC-BY-4.0 — attribution required).
