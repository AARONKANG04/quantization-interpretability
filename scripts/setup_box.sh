#!/usr/bin/env bash
# One-shot environment setup on a fresh GPU box. Run from the repo root after `git clone`.
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev --extra box
set -a; [ -f .env ] && . ./.env; set +a      # HF_TOKEN, WANDB_API_KEY, ANTHROPIC_API_KEY
if [ -n "${HF_TOKEN:-}" ]; then uv run hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || uv run huggingface-cli login --token "$HF_TOKEN" >/dev/null; fi
sudo mkdir -p /data/ckpt /data/kl /data/saes /data/buffers /data/lens 2>/dev/null || mkdir -p /data/ckpt /data/kl /data/saes /data/buffers /data/lens
sudo chown -R "$USER" /data 2>/dev/null || true
uv run hf download google/gemma-3-4b-pt --exclude "*.gguf" >/dev/null 2>&1 || uv run huggingface-cli download google/gemma-3-4b-pt
uv run python scripts/smoke.py
