#!/usr/bin/env bash
# Stage 1 on boxS (one GPU): checkpoints + cross-check in job A, evals chained behind the Q1 build in job B.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/data/hf TOKENIZERS_PARALLELISM=false
set -a; . ./.env; set +a
mkdir -p logs results/phase0/pilot
case "${1:-}" in
  ckpts)
    rm -f logs/q1_ckpt.ready
    uv run python scripts/phase0_pilot.py --scheme none      --out /data/ckpt/bf16 --stats-out results/phase0/pilot/bf16.json && \
    uv run python scripts/phase0_pilot.py --scheme nvfp4_rtn --out /data/ckpt/q1   --stats-out results/phase0/pilot/q1.json && touch logs/q1_ckpt.ready
    uv run python scripts/phase0_pilot.py --scheme fp8_e4m3  --out /data/ckpt/q3   --stats-out results/phase0/pilot/q3.json
    uv run python scripts/phase0_pilot.py --scheme noise_matched --out /data/ckpt/c2 --stats-out results/phase0/pilot/c2.json
    uv run python scripts/phase0_crosscheck.py --out /data/ckpt/q1_llmc
    ;;
  evals)
    while [ ! -e logs/q1_ckpt.ready ]; do sleep 15; done
    uv run python scripts/phase0_eval.py --ckpt /data/ckpt/bf16 --name bf16 --tasks gsm8k_cot,mmlu
    uv run python scripts/phase0_eval.py --ckpt /data/ckpt/q1   --name q1   --tasks gsm8k_cot,mmlu
    ;;
  *) echo "usage: $0 ckpts|evals"; exit 2;;
esac
