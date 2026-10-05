#!/usr/bin/env bash
# One-shot setup of the 8x H100 box: environment, model, headers, deterministic artifacts (token sets, reference
# cache, bf16/q1/c2 checkpoints). Idempotent; safe to rerun. Run from the repo root on the box.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/data/hf TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True UV_PROJECT_ENVIRONMENT=/data/venv
set -a; . ./.env; set +a
mkdir -p logs /data/ckpt /data/kl /data/saes /data/buffers /data/lens /data/hf /data/lm_eval_raw /data/shift /data/saliency /data/selections
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
sudo -n apt-get update -qq >/dev/null 2>&1 || true; sudo -n apt-get install -y -qq python3.10-dev >/dev/null 2>&1 || true
grep -q UV_PROJECT_ENVIRONMENT ~/.bashrc || printf '\nexport PATH="$HOME/.local/bin:$PATH" HF_HOME=/data/hf UV_PROJECT_ENVIRONMENT=/data/venv TOKENIZERS_PARALLELISM=false\n' >> ~/.bashrc
uv sync --extra dev --extra box >/dev/null
uv tool install -q "huggingface_hub[cli]" >/dev/null 2>&1 || true
hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
hf download google/gemma-3-4b-pt --exclude "*.gguf" >/dev/null
uv run pytest -q 2>&1 | tail -1
# token sets (each on its own CPU-bound process, in parallel)
for s in kl_500k kl_2m gsm8k_test lens_10m; do
  [ -e /data/kl/$s.pt ] || (uv run python scripts/data_kl_subset.py --set $s > logs/data_$s.log 2>&1 &)
done
wait
# checkpoints and reference cache, one GPU each
[ -e /data/ckpt/bf16/config.json ] || CUDA_VISIBLE_DEVICES=0 uv run python scripts/phase0_pilot.py --scheme none --out /data/ckpt/bf16 --stats-out results/phase0/pilot/bf16.json > logs/ckpt_bf16.log 2>&1 &
[ -e /data/ckpt/q1/config.json ]   || CUDA_VISIBLE_DEVICES=1 uv run python scripts/phase0_pilot.py --scheme nvfp4_rtn --out /data/ckpt/q1 --stats-out results/phase0/pilot/q1.json > logs/ckpt_q1.log 2>&1 &
[ -e /data/ckpt/c2/config.json ]   || CUDA_VISIBLE_DEVICES=2 uv run python scripts/phase0_pilot.py --scheme noise_matched --out /data/ckpt/c2 --stats-out results/phase0/pilot/c2.json > logs/ckpt_c2.log 2>&1 &
[ -e /data/kl/ref_kl_500k_top256.pt ] || CUDA_VISIBLE_DEVICES=3 uv run python scripts/phase1_layer_sweep.py --build-ref --tokens /data/kl/kl_500k.pt --ref /data/kl/ref_kl_500k_top256.pt --batch 2 > logs/ref_500k.log 2>&1 &
wait
ls -la /data/ckpt /data/kl | head -30
echo "boxA setup done"
