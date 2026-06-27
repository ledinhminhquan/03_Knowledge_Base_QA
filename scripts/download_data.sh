#!/usr/bin/env bash
# Download datasets + build the demo knowledge-base index.
set -euo pipefail
python -m kbqa.cli data --task all
