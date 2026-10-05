#!/usr/bin/env bash
# Everything on boxA: setup, then the job queue with one worker per GPU. usage: scripts/remote/boxA_run.sh [layer]
set -euo pipefail
cd "$(dirname "$0")/../.."
L="${1:-17}"
scripts/remote/boxA_setup.sh
scripts/remote/queue.sh init
scripts/remote/boxA_jobs.sh "$L" > /tmp/jobs.txt
scripts/remote/queue.sh add /tmp/jobs.txt
scripts/remote/queue.sh start-workers "$(nvidia-smi -L | wc -l)"
