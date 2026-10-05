#!/usr/bin/env bash
# Start (or restart after a preemption / box switch) the whole boxA pipeline in a detached tmux session.
# usage: scripts/remote/boxA_launch.sh [layer]
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
tmux has-session -t boxa_run 2>/dev/null && { echo "boxa_run already running"; exit 0; }
tmux new-session -d -s boxa_run "bash -lc 'scripts/remote/boxA_run.sh ${1:-17} >> logs/boxA_run.log 2>&1; echo \"boxA_run exit \$?\" >> logs/boxA_run.log'"
echo "started boxa_run, log: logs/boxA_run.log"
