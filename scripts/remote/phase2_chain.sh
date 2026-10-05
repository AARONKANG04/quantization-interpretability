#!/usr/bin/env bash
# Phase 2 statistics on boxS once the GPU is free: SAE validation, shift accounting (Q1 and C2) at two layers,
# feature selection. Waits for the deferred token dumps to finish first.
cd ~/quantization-interpretability || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/data/hf TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
set -a; . ./.env; set +a
while [ ! -e logs/after_evals.done ] && [ ! -e logs/after_evals.failed ]; do sleep 20; done
for L in 17 29; do
  uv run python scripts/phase2_validate_sae.py --layer $L --ckpt-b /data/ckpt/q1 --n-seqs 48 --out results/phase2/sae_validation_L$L.json
done
for L in 17 29; do
  for cond in q1 c2; do
    uv run python scripts/phase2_shift.py --layer $L --ckpt-b /data/ckpt/$cond --name $cond \
      --tokens /data/kl/kl_2m.pt /data/kl/lens_10m.pt /data/kl/gsm8k_test.pt --batch 4 --out results/phase2/shift
  done
  uv run python scripts/phase2_select_feats.py --shift /data/shift/q1_L$L.pt --out results/phase2/feats_q1_L$L.json
done
