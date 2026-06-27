"""Command-line interface — the single entrypoint for the KBQA system.

    kbqa <command> [options]

Commands: data, build-kb, train-retriever, train-reader, train-generator, tune,
evaluate, ask, search, demo-agent, serve, benchmark, error-analysis, faithfulness,
monitor, generate-report, generate-slides, autopilot, grade.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import AppConfig, ensure_dirs, load_config
from .logging_utils import get_logger

logger = get_logger(__name__)


def _load(args) -> AppConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else AppConfig()
    ensure_dirs()
    return cfg


def cmd_data(args):
    from .data.download_dataset import download_all, download_task
    cfg = _load(args)
    res = download_all(cfg) if args.task == "all" else download_task(args.task, cfg)
    print(json.dumps(res, indent=2))


def cmd_build_kb(args):
    from .data.corpus import build_demo_kb
    from .config import index_dir
    cfg = _load(args)
    r = build_demo_kb(cfg, limit_corpus=args.limit, use_dataset=not args.samples)
    print(json.dumps({"passages": r.size, "index": str(index_dir())}, indent=2))


def cmd_train_retriever(args):
    from .training.train_retriever import train_retriever
    print(json.dumps(train_retriever(_load(args), limit=args.limit), indent=2))


def cmd_train_reader(args):
    from .training.train_reader import train_reader
    print(json.dumps(train_reader(_load(args), limit=args.limit), indent=2))


def cmd_train_generator(args):
    from .training.train_generator import train_generator
    print(json.dumps(train_generator(_load(args), limit=args.limit), indent=2))


def cmd_tune(args):
    from .training.tune import tune_reader
    print(json.dumps(tune_reader(_load(args)), indent=2))


def cmd_evaluate(args):
    from .training.evaluate import evaluate
    print(json.dumps(evaluate(_load(args), limit=args.limit).get("summary", {}), indent=2))


def cmd_ask(args):
    from .agent.rag_agent import RAGAgent
    from .data.samples import SAMPLE_DOCS
    agent = RAGAgent(_load(args))
    if agent.retriever.size == 0:
        agent.ingest(list(SAMPLE_DOCS))
    state = agent.ask(args.question)
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))


def cmd_search(args):
    from .agent.rag_agent import RAGAgent
    from .data.samples import SAMPLE_DOCS
    agent = RAGAgent(_load(args))
    if agent.retriever.size == 0:
        agent.ingest(list(SAMPLE_DOCS))
    cands = agent.retrieve_tool.run(query=args.query, top_k=args.top_k).get("passages", [])
    reranked = agent.rerank_tool.run(query=args.query, candidates=cands, top_n=args.top_k).get("passages", [])
    print(json.dumps(reranked, indent=2, ensure_ascii=False))


def cmd_demo_agent(args):
    from .agent.rag_agent import RAGAgent
    from .data.samples import SAMPLE_DOCS, SAMPLE_QA
    agent = RAGAgent(_load(args))
    agent.ingest(list(SAMPLE_DOCS))
    for qa in SAMPLE_QA:
        state = agent.ask(qa["question"])
        print(f"\nQ: {qa['question']}")
        print(f"  status : {state.status}")
        print(f"  answer : {state.answer}")
        print(f"  conf   : {state.confidence:.2f} faithful={state.faithfulness:.2f}")
        if state.citations:
            print(f"  cite   : {[c.get('title') or c.get('chunk_id') for c in state.citations]}")


def cmd_serve(args):
    import os
    import uvicorn
    if args.config:
        os.environ["KBQA_INFER_CONFIG"] = str(args.config)
    uvicorn.run("kbqa.api.main:app", host=args.host, port=args.port, reload=False)


def cmd_benchmark(args):
    from .analysis.latency import benchmark
    print(json.dumps(benchmark(_load(args), n=args.n, warmup=args.warmup), indent=2))


def cmd_error_analysis(args):
    from .analysis.error_analysis import error_analysis
    print(json.dumps(error_analysis(_load(args), limit=args.limit), indent=2))


def cmd_faithfulness(args):
    from .analysis.faithfulness import faithfulness_eval
    print(json.dumps(faithfulness_eval(_load(args), limit=args.limit), indent=2))


def cmd_monitor(args):
    from .monitoring.drift_report import monitoring_report
    print(json.dumps(monitoring_report(_load(args), log_path=args.log), indent=2))


def cmd_generate_report(args):
    from .autoreport.report_pdf import generate_report
    print("Report ->", generate_report(_load(args), title=args.title, author=args.author))


def cmd_generate_slides(args):
    from .autoreport.slides_pptx import generate_slides
    print("Slides ->", generate_slides(_load(args), title=args.title, author=args.author))


def cmd_autopilot(args):
    from .automation.autopilot import run_autopilot
    res = run_autopilot(_load(args), title=args.title, author=args.author,
                        train=not args.no_train, limit=args.limit)
    print(json.dumps(res, indent=2))


def cmd_grade(args):
    from .grading.checklist import build_checklist
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[2]
    print(json.dumps(build_checklist(repo), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kbqa", description="Knowledge Base QA System (RAG)")
    p.add_argument("--config", help="Path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("data", help="download datasets + build demo KB"); sp.add_argument("--task", choices=["all", "reader", "retriever", "corpus"], default="all"); sp.set_defaults(func=cmd_data)
    sp = sub.add_parser("build-kb", help="build the demo knowledge-base index"); sp.add_argument("--limit", type=int, default=None); sp.add_argument("--samples", action="store_true", help="use built-in samples instead of the dataset"); sp.set_defaults(func=cmd_build_kb)
    sp = sub.add_parser("train-retriever", help="fine-tune the dense retriever"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_train_retriever)
    sp = sub.add_parser("train-reader", help="fine-tune the extractive reader"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_train_reader)
    sp = sub.add_parser("train-generator", help="fine-tune the FLAN-T5 generative reader"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_train_generator)
    sp = sub.add_parser("tune", help="hyperparameter tuning (reader)"); sp.set_defaults(func=cmd_tune)
    sp = sub.add_parser("evaluate", help="retrieval + reader + e2e evaluation"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("ask", help="ask one question"); sp.add_argument("--question", required=True); sp.set_defaults(func=cmd_ask)
    sp = sub.add_parser("search", help="retrieve+rerank passages"); sp.add_argument("--query", required=True); sp.add_argument("--top-k", type=int, default=5); sp.set_defaults(func=cmd_search)
    sp = sub.add_parser("demo-agent", help="run the agent on built-in samples"); sp.set_defaults(func=cmd_demo_agent)
    sp = sub.add_parser("serve", help="start the FastAPI server"); sp.add_argument("--host", default="0.0.0.0"); sp.add_argument("--port", type=int, default=8000); sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("benchmark", help="latency benchmark p50/p95/p99"); sp.add_argument("--n", type=int, default=50); sp.add_argument("--warmup", type=int, default=5); sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("error-analysis", help="QA error analysis"); sp.add_argument("--limit", type=int, default=200); sp.set_defaults(func=cmd_error_analysis)
    sp = sub.add_parser("faithfulness", help="faithfulness/groundedness evaluation"); sp.add_argument("--limit", type=int, default=100); sp.set_defaults(func=cmd_faithfulness)
    sp = sub.add_parser("monitor", help="monitoring report from query logs"); sp.add_argument("--log", default=None); sp.set_defaults(func=cmd_monitor)
    sp = sub.add_parser("generate-report", help="generate the PDF report"); sp.add_argument("--title", default="Knowledge Base Question-Answering System"); sp.add_argument("--author", default="Le Dinh Minh Quan"); sp.set_defaults(func=cmd_generate_report)
    sp = sub.add_parser("generate-slides", help="generate the PPTX slides"); sp.add_argument("--title", default="Knowledge Base Question-Answering System"); sp.add_argument("--author", default="Le Dinh Minh Quan"); sp.set_defaults(func=cmd_generate_slides)
    sp = sub.add_parser("autopilot", help="one-button: build KB -> train -> eval -> analysis -> report+slides"); sp.add_argument("--title", default="Knowledge Base Question-Answering System"); sp.add_argument("--author", default="Le Dinh Minh Quan"); sp.add_argument("--no-train", action="store_true"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_autopilot)
    sp = sub.add_parser("grade", help="rubric completeness self-check"); sp.add_argument("--repo", default=None); sp.set_defaults(func=cmd_grade)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
