#!/usr/bin/env bash
# Minimal GPU job queue. Jobs live in $Q/jobs.txt as `name|dep1,dep2|command`, in priority order.
# One worker per GPU claims the first unclaimed job whose dependencies are done (atomic mkdir), runs it with
# CUDA_VISIBLE_DEVICES set, and marks done/ or failed/. A job whose dependency failed is marked failed.
# usage: queue.sh init | add <jobs-file> | worker <gpu> | status | start-workers <n>
Q="${Q:-/data/queue}"
cmd="${1:-status}"; shift || true
case "$cmd" in
  init) rm -rf "$Q"; mkdir -p "$Q/started" "$Q/done" "$Q/failed" "$Q/logs"; : > "$Q/jobs.txt"; echo "queue at $Q";;
  add) cat "$1" >> "$Q/jobs.txt"; echo "$(grep -c . "$Q/jobs.txt") jobs queued";;
  done) touch "$Q/done/$1"; echo "marked $1 done";;
  requeue-started) for s in "$Q"/started/*; do n=$(basename "$s"); [ -e "$Q/done/$n" ] || [ -e "$Q/failed/$n" ] || { rmdir "$s" && echo "requeued $n"; }; done;;
  retry-failed) for f in "$Q"/failed/*; do n=$(basename "$f"); rm -f "$f"; rmdir "$Q/started/$n" 2>/dev/null; echo "retrying $n"; done;;
  start-workers) n="${1:-8}"; for g in $(seq 0 $((n - 1))); do
      tmux new-session -d -s "worker$g" "bash -lc 'cd $(pwd) && Q=$Q scripts/remote/queue.sh worker $g'"; done; tmux ls;;
  worker)
    gpu="$1"; cd "$(dirname "$0")/../.." || exit 1
    export PATH="$HOME/.local/bin:$PATH" HF_HOME=/data/hf TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True UV_PROJECT_ENVIRONMENT=/data/venv
    set -a; . ./.env; set +a
    while true; do
      claimed=""
      while IFS='|' read -r name deps command; do
        [ -z "$name" ] && continue
        [ -e "$Q/done/$name" ] || [ -e "$Q/failed/$name" ] || [ -d "$Q/started/$name" ] && continue
        ready=1
        IFS=',' read -ra dl <<< "$deps"
        for d in "${dl[@]}"; do
          [ -z "$d" ] && continue
          if [ -e "$Q/failed/$d" ]; then touch "$Q/failed/$name"; echo "$name: dependency $d failed" >> "$Q/logs/$name.log"; ready=0; break; fi
          [ -e "$Q/done/$d" ] || { ready=0; break; }
        done
        [ "$ready" = 1 ] || continue
        mkdir "$Q/started/$name" 2>/dev/null || continue
        claimed="$name"
        echo "[$(date +%H:%M:%S)] gpu$gpu start $name" | tee -a "$Q/logs/_workers.log"
        if CUDA_VISIBLE_DEVICES=$gpu bash -c "$command" > "$Q/logs/$name.log" 2>&1; then touch "$Q/done/$name"; st=done; else touch "$Q/failed/$name"; st=FAILED; fi
        echo "[$(date +%H:%M:%S)] gpu$gpu $st $name" | tee -a "$Q/logs/_workers.log"
        break
      done < "$Q/jobs.txt"
      if [ -z "$claimed" ]; then
        total=$(grep -c . "$Q/jobs.txt"); fin=$(( $(ls "$Q/done" | wc -l) + $(ls "$Q/failed" | wc -l) ))
        [ "$fin" -ge "$total" ] && { echo "[$(date +%H:%M:%S)] gpu$gpu queue drained" | tee -a "$Q/logs/_workers.log"; exit 0; }
        sleep 20
      fi
    done;;
  status)
    total=$(grep -c . "$Q/jobs.txt" 2>/dev/null || echo 0)
    echo "jobs: $total  done: $(ls $Q/done 2>/dev/null | wc -l)  failed: $(ls $Q/failed 2>/dev/null | wc -l)  running: $(for s in $Q/started/*; do n=$(basename $s); [ -e $Q/done/$n ] || [ -e $Q/failed/$n ] || echo $n; done 2>/dev/null | tr '\n' ' ')"
    [ -d "$Q/failed" ] && [ "$(ls $Q/failed | wc -l)" -gt 0 ] && echo "FAILED: $(ls $Q/failed | tr '\n' ' ')"
    tail -n 8 "$Q/logs/_workers.log" 2>/dev/null
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null;;
  *) echo "usage: queue.sh init|add <file>|done <name>|requeue-started|retry-failed|worker <gpu>|start-workers <n>|status"; exit 2;;
esac
