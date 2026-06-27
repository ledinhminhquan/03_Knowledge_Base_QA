#!/usr/bin/env bash
# Build KB, fine-tune retriever + reader, then evaluate.
set -euo pipefail
CONFIG="${1:-configs/train.yaml}"
python -m kbqa.cli build-kb         --config "$CONFIG"
python -m kbqa.cli train-retriever  --config "$CONFIG"
python -m kbqa.cli train-reader     --config "$CONFIG"
python -m kbqa.cli evaluate         --config "$CONFIG"
