#!/usr/bin/env bash
# Pull results/ (small JSON, parquet, figures) back from a box. usage: scripts/remote/sync.sh boxA
set -euo pipefail
host="${1:-boxA}"
rsync -avz --exclude 'samples_*.jsonl' "$host:$(basename "$(pwd)")/results/" results/
echo "synced results/ from $host"
