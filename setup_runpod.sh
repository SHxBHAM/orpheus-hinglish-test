#!/usr/bin/env bash
# One-time setup on a RunPod GPU pod (RTX 4090 / A100). Run from this folder.
set -euo pipefail

echo ">>> Python: $(python --version)"

# Use a clean venv so vLLM's pins don't fight the base image.
python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

# orpheus-speech first, then pin the vLLM it's known-good with.
pip install orpheus-speech
pip install "vllm==0.7.3"
pip install snac huggingface_hub

# Some checkpoints are gated / need the SNAC codec cached. Log in if you hit 401s:
#   huggingface-cli login

echo ">>> Done. Activate with: source .venv/bin/activate"
echo ">>> Then:  python run_orpheus.py --model SachinTelecmi/Orpheus-tts-hi --subset hinglish"
