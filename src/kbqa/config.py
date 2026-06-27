"""Typed configuration + YAML loader for the KBQA system.

Single source of truth for models, datasets, chunking, agent thresholds and
serving. Loaded from ``configs/*.yaml``; paths come from environment variables so
nothing is hard-coded.

Environment overrides
---------------------
* ``KBQA_ARTIFACTS_DIR`` – base for data/models/index/runs (Drive on Colab)
* ``KBQA_MODEL_DIR``     – trained models
* ``KBQA_INDEX_DIR``     – FAISS knowledge-base index
* ``KBQA_RUN_DIR``       – eval/benchmark/analysis JSON
* ``HF_HOME``            – HuggingFace cache
* ``KBQA_LLM_API_KEY``   – optional key for the LLM brain (never committed)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def artifacts_dir() -> Path:
    return Path(_env("KBQA_ARTIFACTS_DIR", "artifacts")).expanduser()


def data_dir() -> Path:
    return Path(_env("KBQA_DATA_DIR", str(artifacts_dir() / "data"))).expanduser()


def model_dir() -> Path:
    return Path(_env("KBQA_MODEL_DIR", str(artifacts_dir() / "models"))).expanduser()


def index_dir() -> Path:
    return Path(_env("KBQA_INDEX_DIR", str(artifacts_dir() / "index"))).expanduser()


def run_dir() -> Path:
    return Path(_env("KBQA_RUN_DIR", str(artifacts_dir() / "runs"))).expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Verified public datasets (see docs/data_description.md)."""
    reader_dataset: str = "rajpurkar/squad_v2"
    reader_config: str = "squad_v2"
    multihop_dataset: str = "hotpotqa/hotpot_qa"
    retriever_pairs_dataset: str = "sentence-transformers/natural-questions"
    retriever_pairs_config: str = "pair"
    demo_kb_dataset: str = "rag-datasets/rag-mini-wikipedia"
    demo_kb_corpus_config: str = "text-corpus"
    demo_kb_qa_config: str = "question-answer"
    recall_dataset: str = "rag-datasets/rag-mini-bioasq"
    generative_rag_dataset: str = "neural-bridge/rag-dataset-12000"
    seed: int = 42


@dataclass
class ChunkConfig:
    """Document chunking (word windows that fit the encoder's ~512-token cap)."""
    chunk_size_words: int = 256     # ~340 tokens, safely under bge/MiniLM 512 cap
    overlap_words: int = 48
    min_words: int = 24


@dataclass
class RetrieverConfig:
    bi_encoder_model: str = "BAAI/bge-base-en-v1.5"
    bi_encoder_fallback: str = "sentence-transformers/all-MiniLM-L6-v2"
    # bge needs a query-side instruction; symmetric models (MiniLM) must NOT have one.
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    e5_style: bool = False           # set True for intfloat/e5-* (query:/passage: prefixes)
    embed_batch_size: int = 64
    top_k: int = 20                  # dense ANN shortlist
    use_bm25: bool = True            # hybrid BM25 + dense via RRF
    rrf_k: int = 60
    # training
    train_batch_size: int = 64
    num_train_epochs: int = 3
    learning_rate: float = 2.0e-5
    warmup_ratio: float = 0.1
    num_hard_negatives: int = 8
    output_subdir: str = "retriever"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir


@dataclass
class RerankerConfig:
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_gpu: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 5
    use_gpu_reranker: bool = False
    max_length: int = 512


@dataclass
class ReaderConfig:
    """Extractive span reader (abstains via SQuAD2 null score)."""
    model_name: str = "deepset/roberta-base-squad2"
    model_gpu: str = "deepset/deberta-v3-large-squad2"
    max_length: int = 384
    doc_stride: int = 128
    null_score_threshold: float = 0.0   # tuned on dev; >0 favours abstention
    max_answer_length: int = 64
    # training
    num_train_epochs: int = 2
    learning_rate: float = 3.0e-5
    per_device_train_batch_size: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    bf16: bool = True
    fp16: bool = False
    early_stopping_patience: int = 3
    seed: int = 42
    output_subdir: str = "reader"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir


@dataclass
class GeneratorConfig:
    """Optional generative grounded reader (FLAN-T5)."""
    model_name: str = "google/flan-t5-base"
    model_large: str = "google/flan-t5-large"
    max_input_length: int = 1024
    max_target_length: int = 128
    num_train_epochs: int = 3
    learning_rate: float = 1.0e-4
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    label_smoothing_factor: float = 0.1
    bf16: bool = True               # FLAN-T5 NaNs under fp16 — never use fp16
    output_subdir: str = "generator"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir


@dataclass
class AgentConfig:
    reader_mode: str = "extractive"   # "extractive" | "generative"
    max_iterations: int = 3           # CRAG correction-loop bound
    tau_high: float = 0.55            # rerank score >= -> SUFFICIENT
    tau_low: float = 0.15             # rerank score <  -> INSUFFICIENT
    faithfulness_threshold: float = 0.30
    require_citations: bool = True
    orchestrator: str = "rule"        # "rule" | "llm"
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key_env: str = "KBQA_LLM_API_KEY"
    llm_max_tokens: int = 512


@dataclass
class ServingConfig:
    model_version: str = "v1"
    api_title: str = "Knowledge Base QA API"
    api_version: str = "1.0.0"
    log_queries: bool = True
    query_log_subdir: str = "query_logs"
    max_batch_questions: int = 64

    @property
    def query_log_path(self) -> Path:
        return run_dir() / self.query_log_subdir / "queries.jsonl"


@dataclass
class AppConfig:
    project_title: str = "Knowledge Base Question-Answering System"
    author: str = "Le Dinh Minh Quan"
    data: DataConfig = field(default_factory=DataConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "data": DataConfig, "chunk": ChunkConfig, "retriever": RetrieverConfig,
    "reranker": RerankerConfig, "reader": ReaderConfig, "generator": GeneratorConfig,
    "agent": AgentConfig, "serving": ServingConfig,
}


def _build(cls, raw: Optional[Dict[str, Any]]):
    raw = raw or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Optional[str | os.PathLike] = None) -> AppConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    top = {k: raw[k] for k in ("project_title", "author") if k in raw}
    sections = {name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    return AppConfig(**top, **sections)


def save_config(cfg: AppConfig, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_dirs() -> Dict[str, Path]:
    dirs = {"artifacts": artifacts_dir(), "data": data_dir(), "models": model_dir(),
            "index": index_dir(), "runs": run_dir()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


__all__ = ["DataConfig", "ChunkConfig", "RetrieverConfig", "RerankerConfig", "ReaderConfig",
           "GeneratorConfig", "AgentConfig", "ServingConfig", "AppConfig",
           "load_config", "save_config", "ensure_dirs",
           "artifacts_dir", "data_dir", "model_dir", "index_dir", "run_dir"]
