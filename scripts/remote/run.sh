#!/usr/bin/env bash
# Launch a long job on a GPU inside tmux with a log and a done marker.
# usage: scripts/remote/run.sh <name> <gpu-ids> <command...>
#   e.g. scripts/remote/run.sh eval_q1 0 uv run python scripts/phase0_eval.py --ckpt /data/ckpt/q1 --name q1
set -euo pipefail
name="$1"; gpus="$2"; shift 2
mkdir -p logs
rm -f "logs/$name.done" "logs/$name.failed"
cmd="cd $(pwd) && CUDA_VISIBLE_DEVICES=$gpus $* > logs/$name.log 2>&1 && touch logs/$name.done || touch logs/$name.failed"
tmux new-session -d -s "$name" "bash -lc '$cmd'"
echo "started tmux session '$name' on GPU(s) $gpus -> logs/$name.log"
