# `data/` — datasets & download scripts

**Large datasets and corpora are never committed** (see `.gitignore`). This holds
download/preparation scripts plus small samples only.

```bash
# Download + prepare all datasets and build the demo knowledge base
python -m kbqa.cli data --task all

# Individual tasks
python -m kbqa.cli data --task qa         # reader QA dataset (question/context/answer)
python -m kbqa.cli data --task corpus     # demo RAG corpus + QA pairs for the KB
```

Exact dataset identifiers, sizes and licenses are documented in
[`../docs/data_description.md`](../docs/data_description.md). The knowledge-base
corpus is chunked and embedded into a FAISS index at build time (no raw corpus is
committed).
