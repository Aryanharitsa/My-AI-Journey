#!/usr/bin/env bash
# Phase 2 preview — one-shot pod bootstrap.
# Operator sets HF_TOKEN and WANDB_API_KEY before running.
set -euo pipefail

# System
apt-get update -qq
apt-get install -y -qq git htop tmux

# Clone
cd /workspace
[ -d my-ai-journey ] || git clone https://github.com/Aryanharitsa/My-AI-Journey.git my-ai-journey
cd my-ai-journey/vitruvius

# Python env
pip install --upgrade pip
pip install -e ".[dev,pod]"

# Swap faiss-cpu for faiss-gpu
pip uninstall -y faiss-cpu || true
pip install faiss-gpu

# HF / WandB auth
if [ -n "${HF_TOKEN:-}" ]; then huggingface-cli login --token "$HF_TOKEN"; fi
if [ -n "${WANDB_API_KEY:-}" ]; then wandb login "$WANDB_API_KEY"; fi

# Sanity
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m vitruvius.cli smoke --cpu --no-encoder

echo "Pod ready."
