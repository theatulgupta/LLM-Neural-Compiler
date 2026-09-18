#!/usr/bin/env bash
# Create a venv and install this tree. LLM keys stay in ~/.config/nnc/, never git.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python -m compiler emit-fixture --out fixtures/tiny_cnn.onnx
python -m compiler probe
echo "bootstrap ok. copy a key with: scp ~/.config/nnc/llm.env <host>:~/.config/nnc/llm.env"
