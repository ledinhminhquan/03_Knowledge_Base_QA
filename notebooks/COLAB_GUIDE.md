# 🚀 Colab Training Guide (H100 / flexible GPU)

From zero to a trained RAG system + auto-generated report & slides, using
[`KBQA_Colab_Training_H100_AUTOPILOT.ipynb`](KBQA_Colab_Training_H100_AUTOPILOT.ipynb).
You do two things: **(1) put the code on GitHub** and **(2) press Run all.**

---

## Step 1 — Put the project on GitHub (one time)

```bash
cd "03_Knowledge_Base_QA"
git init && git add . && git commit -m "Knowledge Base QA System"
git branch -M main
git remote add origin https://github.com/<your-username>/kbqa.git
git push -u origin main
```

> `models/`, `artifacts/`, `index/` and large data are git-ignored.
> **Alternative (no GitHub):** upload the folder to Drive and set `DRIVE_REPO_DIR`.

---

## Step 2 — Google Drive layout (auto-created)

On first run the notebook creates this under your Drive and **persists all
artifacts** across sessions:

```
MyDrive/
└── NLP_Project/
    └── kbqa/
        ├── hf_cache/                 # HuggingFace cache (survives disconnects)
        └── artifacts/
            ├── data/
            ├── index/                # FAISS knowledge-base index
            ├── models/
            │   ├── retriever/latest/  # fine-tuned bi-encoder
            │   └── reader/latest/     # fine-tuned extractive reader
            ├── runs/                  # eval / benchmark / analysis JSON
            └── submission/
                └── submission-<timestamp>/  # report.pdf, slides.pptx, bundle.zip
```

`DRIVE_PROJECT_DIR` (default `NLP_Project/kbqa`) sets this path.

---

## Step 3 — Open in Colab and run

1. Upload the notebook to Colab (or open from Drive/GitHub).
2. **Runtime → Change runtime type → GPU.** Prefer **H100**; if unavailable pick
   **A100 / L4 / T4** — the notebook auto-adapts batch size + precision.
3. In **Controls** (cell 0): set `GIT_REPO_URL`, keep `RETRIEVER_MODEL =
   BAAI/bge-base-en-v1.5` and `READER_MODEL = deepset/roberta-base-squad2`
   (defaults). Leave `RUN_AUTOPILOT = True`.
4. **Runtime → Run all.**

The autopilot (cell 10): build KB → fine-tune retriever (MNRL + hard negatives)
→ fine-tune reader (SQuAD v2) → rebuild index → evaluate → benchmark → error
analysis → faithfulness → **generate `report.pdf` + `slides.pptx`** + grading
checklist.

⏱️ On H100, the demo KB + a capped fine-tune is minutes; full SQuAD v2 reader
training is ~30–60 min (use `DEBUG_LIMIT = 2000` for a fast first pass).

---

## Step 4 — If Colab disconnects (no work lost)

Reconnect → **Runtime → Run all** again. Training **resumes from the last
checkpoint** on Drive; completed steps are skipped.

---

## Step 5 — Test the trained model

Cell 13 loads the fine-tuned retriever + reader + KB index and asks grounded
questions — including one **not** in the KB, which the agent should **abstain** on
("I don't know") instead of hallucinating.

Add your own documents and ask about them:

```python
from kbqa.config import load_config
from kbqa.agent.rag_agent import RAGAgent
agent = RAGAgent(load_config("configs/train_colab.yaml"))
agent.ingest([{"title": "My notes", "text": "Our product launches in March 2027."}])
print(agent.ask("When does the product launch?").to_dict())
```

---

## Step 6 — Collect your deliverables

Cell 14 prints the submission folder. Download from Drive:

```
artifacts/submission/submission-<timestamp>/
├── report.pdf            # 10–15 page report (auto-generated from docs + metrics)
├── slides.pptx           # 12-slide deck
├── grading_checklist.json
└── submission_bundle.zip
```

Submit the **GitHub link**, the **report.pdf** and the **slides.pptx**.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Set GIT_REPO_URL ...` in cell 4 | Fill `GIT_REPO_URL` (or `DRIVE_REPO_DIR`) in Controls. |
| H100 unavailable | Pick A100/L4/T4 — the notebook adapts automatically. |
| OOM training the reader | Use `deepset/roberta-base-squad2` (not -large), or lower `DEBUG_LIMIT`. |
| Reader training slow | Set `DEBUG_LIMIT = 2000` for a fast pass; raise later for the final run. |
| Agent answers from memory | It shouldn't — the faithfulness gate forces abstention; check `agent.require_citations=True`. |
| Want generative answers | Set `agent.reader_mode: generative` and run `kbqa train-generator` (FLAN-T5). |
