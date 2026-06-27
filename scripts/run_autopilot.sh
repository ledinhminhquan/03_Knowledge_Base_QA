#!/usr/bin/env bash
# One button: build KB -> train -> evaluate -> analysis -> report + slides.
set -euo pipefail
python -m kbqa.cli autopilot \
  --config "${1:-configs/train.yaml}" \
  --title "Knowledge Base Question-Answering System" \
  --author "Le Dinh Minh Quan"
