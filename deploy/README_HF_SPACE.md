# Deploying to a Hugging Face Space (Docker SDK)

The KBQA system serves a combined **REST API + Gradio demo** in one process on
port **7860** — exactly what a Hugging Face *Docker* Space expects.

## Steps

1. Create a Space → SDK = **Docker** → CPU hardware is fine (GPU optional).
2. Push this repository to the Space repo. Add this YAML block to the top of the
   Space's `README.md` (HF reads it as the Space config):

```yaml
---
title: Knowledge Base QA System
emoji: 📚
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---
```

3. The Space builds the `Dockerfile` and starts:
   `uvicorn kbqa.api.app_combined:app --host 0.0.0.0 --port 7860`

4. Once live:
   - Demo UI: `https://<user>-<space>.hf.space/ui`
   - API docs: `https://<user>-<space>.hf.space/docs`
   - Health:   `https://<user>-<space>.hf.space/health`

## Notes

* Default deployment needs **no secrets** and runs the rule-based agent on CPU.
  The demo seeds a small built-in KB; use `POST /ingest` to add your own documents.
* To ship a prebuilt FAISS index, upload `artifacts/index/` (or build it at
  startup) and set `KBQA_INDEX_DIR`.
* Enable the optional LLM brain with a Space secret `KBQA_LLM_API_KEY` and
  `agent.orchestrator: "llm"`.
