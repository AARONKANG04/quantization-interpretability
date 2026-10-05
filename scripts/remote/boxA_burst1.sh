#!/usr/bin/env bash
# Burst 1 on the 8x H100 box. Lanes:
#   GPU 0: self-trained SAE on bf16 activations, layer $L      GPU 1: self-trained SAE on Q1 activations, layer $L
#   GPU 2: tuned lens (train on lens_10m, apply to Q1)         GPU 3: build the 10 ranked-curve checkpoints, then eval 2 of them
#   GPUs 4-7: evaluate the remaining ranked-curve checkpoints (GSM8K full + MMLU 70 items/subtask)
# usage: scripts/remote/boxA_burst1.sh [layer]   (needs results/phase1/ranked_order.json synced from boxS)
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/data/hf TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
set -a; . ./.env; set +a
L="${1:-17}"
R=scripts/remote/run.sh
$R sae_bf16_L$L 0 uv run python scripts/phase2_train_sae.py --model-ckpt /data/ckpt/bf16 --name bf16 --layer $L --tokens 100000000
$R sae_q1_L$L   1 uv run python scripts/phase2_train_sae.py --model-ckpt /data/ckpt/q1   --name q1   --layer $L --tokens 100000000
$R lens 2 bash -c "\"uv run python scripts/phase1_tuned_lens.py train --tokens /data/kl/lens_10m.pt --out /data/lens/gemma-3-4b-pt.pt && uv run python scripts/phase1_tuned_lens.py apply --lens /data/lens/gemma-3-4b-pt.pt --ckpt-b /data/ckpt/q1 --tokens /data/kl/kl_500k.pt --out results/phase1/lens_q1.json\""
$R ranked_build 3 uv run python scripts/phase1_ranked_curve.py --sweep results/phase1/layer_sweep --out results/phase1/ranked_order.json
# evals start once the checkpoints exist: one waiter per GPU, each takes its share of the 10 checkpoints
cat > logs/ranked_eval_lane.sh <<'EOS'
#!/usr/bin/env bash
cd ~/quantization-interpretability; export PATH="$HOME/.local/bin:$PATH" HF_HOME=/data/hf TOKENIZERS_PARALLELISM=false
set -a; . ./.env; set +a
while [ ! -e logs/ranked_build.done ]; do sleep 30; done
for name in "$@"; do
  uv run python scripts/phase0_eval.py --ckpt /data/ckpt/$name --name rc_$name --tasks gsm8k_cot --gpu-util 0.8
  uv run python scripts/phase0_eval.py --ckpt /data/ckpt/$name --name rc_${name}_mmlu --tasks mmlu --limit 70 --gpu-util 0.8
done
EOS
chmod +x logs/ranked_eval_lane.sh
$R rc_lane3 3 logs/ranked_eval_lane.sh ranked_top1 random_s0_top1
$R rc_lane4 4 logs/ranked_eval_lane.sh ranked_top2 random_s0_top2
$R rc_lane5 5 logs/ranked_eval_lane.sh ranked_top3 random_s0_top3
$R rc_lane6 6 logs/ranked_eval_lane.sh ranked_top5 random_s0_top5
$R rc_lane7 7 logs/ranked_eval_lane.sh ranked_top8 random_s0_top8
tmux ls
# layer-17 Q1 shift accounting re-run with the mean-shift / top-direction measurement, after lane 7's evals
$R shift_q1_L17_v2 7 bash -c "\"while [ ! -e logs/rc_lane7.done ] && [ ! -e logs/rc_lane7.failed ]; do sleep 30; done; uv run python scripts/phase2_shift.py --layer $L --ckpt-b /data/ckpt/q1 --name q1v2 --tokens /data/kl/kl_2m.pt /data/kl/lens_10m.pt /data/kl/gsm8k_test.pt --batch 4 --out results/phase2/shift\""
