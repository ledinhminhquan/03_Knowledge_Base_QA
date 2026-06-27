# Knowledge Base QA System — container image (REST API + Gradio demo, port 7860).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    KBQA_ARTIFACTS_DIR=/app/artifacts \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY app ./app
COPY docs ./docs
RUN pip install -e .

# Pre-download the small CPU fallback encoder so cold start is offline-fast.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')" || true

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["uvicorn", "kbqa.api.app_combined:app", "--host", "0.0.0.0", "--port", "7860"]
