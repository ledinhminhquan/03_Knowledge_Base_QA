"""Two interchangeable agent brains for the three decision points.

* :class:`RuleOrchestrator` — deterministic heuristics (:mod:`kbqa.agent.policy`),
  zero external dependency, the default.
* :class:`LLMOrchestrator` — optional Anthropic brain for query decomposition,
  sufficiency (Self-RAG ISREL) and faithfulness (Self-RAG ISSUP). It validates
  every response and **falls back to the rule version** on any error.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

from ..config import AgentConfig
from ..logging_utils import get_logger
from . import policy

logger = get_logger(__name__)


class RuleOrchestrator:
    name = "rule"

    def analyze(self, question: str, cfg: AgentConfig) -> Dict:
        return policy.analyze_query(question)

    def sufficiency(self, question: str, reranked: Sequence[Dict], cfg: AgentConfig) -> Dict:
        return policy.assess_sufficiency(question, reranked, cfg)

    def faithfulness(self, answer: str, passages: Sequence[Dict], cfg: AgentConfig, encoder=None) -> Dict:
        return policy.assess_faithfulness(answer, passages, cfg, encoder=encoder)

    def clarify(self, question: str, missing: Optional[List[str]] = None) -> str:
        return policy.make_clarifying_question(question, missing)


class LLMOrchestrator(RuleOrchestrator):
    name = "llm"

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._client = None
        self._enabled = True
        key = os.environ.get(cfg.llm_api_key_env)
        if not key:
            logger.info("No LLM key in $%s; LLMOrchestrator falls back to rules.", cfg.llm_api_key_env)
            self._enabled = False
        else:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=key)
            except Exception as exc:
                logger.warning("anthropic unavailable (%s); rule fallback.", exc)
                self._enabled = False

    def _call(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.cfg.llm_model, max_tokens=self.cfg.llm_max_tokens,
            system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def analyze(self, question: str, cfg: AgentConfig) -> Dict:
        if not self._enabled:
            return super().analyze(question, cfg)
        try:
            sys_p = ("Decompose a question for retrieval. Return JSON "
                     "{\"qtype\":\"simple|multihop|unanswerable\",\"is_multi_hop\":bool,"
                     "\"rewritten\":str,\"sub_questions\":[str]}.")
            raw = self._call(sys_p, f"Question: {question}")
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            data.setdefault("sub_questions", [])
            data.setdefault("rewritten", question)
            return data
        except Exception as exc:
            logger.warning("LLM analyze failed (%s); rule fallback.", exc)
            return super().analyze(question, cfg)

    def sufficiency(self, question: str, reranked: Sequence[Dict], cfg: AgentConfig) -> Dict:
        if not self._enabled or not reranked:
            return super().sufficiency(question, reranked, cfg)
        try:
            ctx = "\n".join(f"- {c['text'][:300]}" for c in reranked[:3])
            sys_p = ("Decide if the context can answer the question. Return JSON "
                     "{\"verdict\":\"SUFFICIENT|AMBIGUOUS|INSUFFICIENT\",\"score\":0..1,\"missing_terms\":[str]}.")
            raw = self._call(sys_p, f"Question: {question}\nContext:\n{ctx}")
            return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except Exception as exc:
            logger.warning("LLM sufficiency failed (%s); rule fallback.", exc)
            return super().sufficiency(question, reranked, cfg)

    def faithfulness(self, answer: str, passages: Sequence[Dict], cfg: AgentConfig, encoder=None) -> Dict:
        if not self._enabled or not answer:
            return super().faithfulness(answer, passages, cfg, encoder)
        try:
            ctx = "\n".join(f"- {p['text'][:300]}" for p in passages[:3])
            sys_p = ("Is the answer fully supported by the context? Return JSON "
                     "{\"supported\":bool,\"support_score\":0..1}.")
            raw = self._call(sys_p, f"Answer: {answer}\nContext:\n{ctx}")
            return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except Exception as exc:
            logger.warning("LLM faithfulness failed (%s); rule fallback.", exc)
            return super().faithfulness(answer, passages, cfg, encoder)


def make_orchestrator(cfg: AgentConfig):
    return LLMOrchestrator(cfg) if cfg.orchestrator == "llm" else RuleOrchestrator()


__all__ = ["RuleOrchestrator", "LLMOrchestrator", "make_orchestrator"]
