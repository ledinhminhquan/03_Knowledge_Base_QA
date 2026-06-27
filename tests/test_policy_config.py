from kbqa.config import AppConfig, load_config, save_config
from kbqa.agent import policy
from kbqa.agent.state import AgentState, AnswerStatus


def test_default_config():
    cfg = AppConfig()
    assert cfg.retriever.bi_encoder_model == "BAAI/bge-base-en-v1.5"
    assert cfg.reader.model_name == "deepset/roberta-base-squad2"
    assert cfg.agent.orchestrator == "rule"


def test_config_roundtrip(tmp_path):
    cfg = AppConfig()
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.agent.tau_high == cfg.agent.tau_high
    assert loaded.reader.doc_stride == cfg.reader.doc_stride


def test_analyze_query_detects_multihop():
    simple = policy.analyze_query("What does FAISS stand for?")
    assert simple["qtype"] == "simple"
    multi = policy.analyze_query("Who founded SpaceX and which university did the founder attend?")
    assert multi["is_multi_hop"] is True
    assert len(multi["sub_questions"]) >= 2


def test_assess_faithfulness_lexical():
    cfg = AppConfig().agent
    passages = [{"text": "FAISS stands for Facebook AI Similarity Search."}]
    good = policy.assess_faithfulness("Facebook AI Similarity Search", passages, cfg)
    bad = policy.assess_faithfulness("The capital of France is Paris", passages, cfg)
    assert good["support_score"] > bad["support_score"]


def test_sufficiency_verdicts():
    cfg = AppConfig().agent
    none = policy.assess_sufficiency("q", [], cfg)
    assert none["verdict"] == "INSUFFICIENT"


def test_agent_state_serialisable():
    s = AgentState(question="q")
    s.status = AnswerStatus.ANSWERED
    d = s.to_dict()
    assert d["status"] == "answered"
    assert "trace" in d
