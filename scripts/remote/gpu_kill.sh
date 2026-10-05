#!/usr/bin/env bash
# Kill every compute process on the given GPU index (default 0) by PID, no name patterns. usage: gpu_kill.sh [gpu]
gpu="${1:-0}"
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' ')
if [ -z "$pids" ]; then echo "GPU $gpu: no compute processes"; exit 0; fi
for p in $pids; do echo "killing pid $p ($(ps -o comm= -p "$p" 2>/dev/null))"; kill "$p" 2>/dev/null; done
sleep 3
for p in $pids; do kill -9 "$p" 2>/dev/null; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$gpu"
