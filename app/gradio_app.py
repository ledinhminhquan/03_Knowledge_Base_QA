"""Local launcher for the Gradio demo.

    python app/gradio_app.py            # serves the demo on http://localhost:7860

For a combined REST API + UI (Hugging Face Space style):
    uvicorn kbqa.api.app_combined:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kbqa.api.ui import launch  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    launch(server_name="0.0.0.0", server_port=port)
