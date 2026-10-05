#!/usr/bin/env bash
# Remaining Phase 0 evals after the Q1 run: fp32 reference (C1), FP8 (Q3), ARC and HellaSwag for bf16 and Q1.
# Runs at a lower vLLM memory share so the Phase 2 chain can share the GPU.
cd ~/quantization-interpretability || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/data/hf TOKENIZERS_PARALLELISM=false
set -a; . ./.env; set +a
while [ ! -e logs/stage1_evals3.done ] && [ ! -e logs/stage1_evals3.failed ]; do sleep 20; done
while [ ! -e logs/sweep_restore.done ] && [ ! -e logs/sweep_restore.failed ]; do sleep 20; done
U=0.45
uv run python scripts/phase0_eval.py --ckpt /data/ckpt/q3   --name q3       --tasks gsm8k_cot,mmlu --gpu-util $U
uv run python scripts/phase0_eval.py --ckpt /data/ckpt/bf16 --name c1_fp32  --tasks gsm8k_cot,mmlu --dtype float32 --gpu-util $U
uv run python scripts/phase0_eval.py --ckpt /data/ckpt/bf16 --name bf16_arc_hs --tasks arc_challenge,hellaswag --gpu-util $U
uv run python scripts/phase0_eval.py --ckpt /data/ckpt/q1   --name q1_arc_hs   --tasks arc_challenge,hellaswag --gpu-util $U
uv run python scripts/phase0_eval.py --ckpt /data/ckpt/q3   --name q3_arc_hs   --tasks arc_challenge,hellaswag --gpu-util $U
