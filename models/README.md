# `models/` — trained models & checkpoints

**Nothing here is committed to Git** (see `.gitignore`). Training writes here (or to
a Drive/`ARTIFACTS_DIR` location on Colab); inference loads from `KBQA_MODEL_DIR`.

If a fine-tuned model is absent, the system falls back to the pretrained retriever /
reader so the demo always runs.

## Expected layout after training

```
models/
├── retriever/latest/      # fine-tuned bi-encoder (sentence-transformers) + model_metadata.json
├── reader/latest/         # fine-tuned extractive QA model (+ label/version metadata)
├── generator/latest/      # (optional) fine-tuned FLAN-T5 grounded reader
└── README.md
```

The FAISS knowledge-base index is stored separately under `artifacts/index/` (see
`KBQA_INDEX_DIR`), not here.
