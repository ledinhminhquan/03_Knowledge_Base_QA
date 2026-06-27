"""Gradio demo UI for the KBQA system (mounted alongside the FastAPI app)."""

from __future__ import annotations

from typing import Optional

from ..agent.rag_agent import RAGAgent, get_agent
from ..config import AppConfig
from ..data.samples import SAMPLE_DOCS
from ..logging_utils import get_logger

logger = get_logger(__name__)

_STATUS_BADGE = {
    "answered": "🟢 ANSWERED",
    "insufficient": "🟠 NOT ENOUGH INFORMATION (abstained)",
    "no_answer": "⚪ NO ANSWER",
    "needs_clarification": "🟡 NEEDS CLARIFICATION",
}


def build_demo(agent: Optional[RAGAgent] = None):
    import gradio as gr

    agent = agent or get_agent(AppConfig())
    # Seed the demo KB with the built-in samples if the index is empty.
    if agent.retriever.size == 0:
        try:
            agent.ingest(list(SAMPLE_DOCS))
        except Exception as exc:
            logger.warning("Could not seed demo KB: %s", exc)

    def ask_fn(question):
        if not question.strip():
            return "Ask a question about the knowledge base.", "", "", []
        state = agent.ask(question)
        d = state.to_dict()
        badge = _STATUS_BADGE.get(str(d.get("status")), str(d.get("status")))
        header = f"### {badge}\n\n**Answer:** {d.get('answer','')}\n\n" \
                 f"**Confidence:** {d.get('confidence',0):.2f} · **Faithfulness:** {d.get('faithfulness',0):.2f}"
        cites = "\n".join(
            f"- **{c.get('title') or c.get('doc_id') or c.get('chunk_id')}**: {c.get('quote','')[:160]}"
            for c in d.get("citations", [])) or "_no citations (abstained)_"
        return header, "\n".join(d.get("clarifying_questions", [])) or "—", cites, d.get("trace", [])

    def ingest_fn(text, title):
        if not text.strip():
            return "Paste some document text to add to the knowledge base."
        out = agent.ingest([{"text": text, "title": title or "user_doc"}])
        return f"Ingested 1 document → {out['new_chunks']} new chunks. KB now has {out['index_n_vectors']} passages."

    with gr.Blocks(title="Knowledge Base QA") as demo:
        gr.Markdown("# 📚 Knowledge Base Question-Answering (RAG)\n"
                    "Ask questions answered **only** from the knowledge base, with citations. "
                    "The agent abstains (\"I don't know\") instead of hallucinating.")
        with gr.Tab("Ask"):
            q_in = gr.Textbox(label="Your question", lines=2,
                              value="What does FAISS stand for?")
            ask_btn = gr.Button("Ask", variant="primary")
            out_ans = gr.Markdown()
            out_clarify = gr.Textbox(label="Clarifying questions", lines=2)
            out_cites = gr.Markdown(label="Citations")
            out_trace = gr.JSON(label="Agent trace (retrieve → rerank → sufficiency → generate → faithfulness)")
            ask_btn.click(ask_fn, [q_in], [out_ans, out_clarify, out_cites, out_trace])
        with gr.Tab("Add to knowledge base"):
            doc_in = gr.Textbox(label="Document text", lines=8)
            title_in = gr.Textbox(label="Title (optional)")
            ing_btn = gr.Button("Ingest", variant="primary")
            ing_out = gr.Markdown()
            ing_btn.click(ingest_fn, [doc_in, title_in], [ing_out])
        gr.Markdown("> Answers are grounded in retrieved passages and cite their sources. "
                    "If the KB lacks the answer, the system says so rather than guessing.")
    return demo


def launch(server_name: str = "0.0.0.0", server_port: int = 7860):
    build_demo().queue(max_size=32).launch(server_name=server_name, server_port=server_port)


__all__ = ["build_demo", "launch"]
