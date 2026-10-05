#!/usr/bin/env bash
# Show running tmux jobs, GPU memory, and the last lines of each active log.
set -uo pipefail
echo "== tmux sessions =="; tmux ls 2>/dev/null || echo "(none)"
echo; echo "== GPUs =="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
echo; echo "== jobs =="
for f in logs/*.log; do
  [ -e "$f" ] || continue
  n=$(basename "$f" .log)
  if [ -e "logs/$n.done" ]; then st=DONE; elif [ -e "logs/$n.failed" ]; then st=FAILED; else st=running; fi
  echo "-- $n [$st]"; tail -n ${TAIL:-3} "$f" | sed 's/^/   /'
done
