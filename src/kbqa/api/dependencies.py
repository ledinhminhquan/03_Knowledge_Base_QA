"""Shared API dependencies: the RAG-agent singleton + config loading."""

from __future__ import annotations

import os
from functools import lru_cache

from ..agent.rag_agent import RAGAgent, get_agent
from ..config import AppConfig, load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    path = os.environ.get("KBQA_INFER_CONFIG")
    if path and os.path.exists(path):
        logger.info("Loading config from %s", path)
        return load_config(path)
    return AppConfig()


def get_rag_agent() -> RAGAgent:
    return get_agent(get_config())


__all__ = ["get_config", "get_rag_agent"]
