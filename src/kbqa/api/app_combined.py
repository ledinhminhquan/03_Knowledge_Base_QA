"""Combined FastAPI + Gradio app (HF Space, port 7860).

Run: uvicorn kbqa.api.app_combined:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

from ..logging_utils import get_logger
from .main import app as fastapi_app

logger = get_logger(__name__)


def _mount_ui(base_app):
    try:
        import gradio as gr
        from .ui import build_demo
        return gr.mount_gradio_app(base_app, build_demo(), path="/ui")
    except Exception as exc:  # pragma: no cover
        logger.warning("Gradio UI not mounted (%s); REST API still available.", exc)
        return base_app


app = _mount_ui(fastapi_app)


@fastapi_app.get("/")
def _root():
    return {"message": "Knowledge Base QA System", "api_docs": "/docs", "demo_ui": "/ui", "health": "/health"}


__all__ = ["app"]
