#!/usr/bin/env bash
# Kill any running token-set builder and relaunch it detached. usage: scripts/remote/build_sets.sh [set ...]
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/data/hf
sets="${*:-kl_500k gsm8k_test kl_2m lens_10m pg19_long}"
for pid in $(pgrep -f "python scripts/data_kl_subset.py"); do kill "$pid" 2>/dev/null; done
sleep 1
setsid nohup bash -c "for s in $sets; do uv run python scripts/data_kl_subset.py --set \$s || echo FAILED \$s; done" > logs/data_sets.log 2>&1 < /dev/null &
echo "builder relaunched for: $sets"
