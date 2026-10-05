#!/usr/bin/env bash
# Pull the durable artifacts of a box into the local archive (incremental, safe to rerun before the box is stopped).
# SAEs (JumpReLU + BatchTopK training checkpoints), tuned lens, shift tensors, saliency scores, channel selections,
# raw lm-eval samples. QDQ and mixed checkpoints are not pulled: phase0_pilot.py / phase3_build.py regenerate them
# deterministically from the selections in a few minutes.
# usage: scripts/remote/archive_pull.sh [box=boxA] [dest=~/qi_archive/<box>]
box="${1:-boxA}"; dest="${2:-$HOME/qi_archive/$box}"
mkdir -p "$dest"
rsync -az --partial "$box":/data/saes "$box":/data/lens "$box":/data/shift "$box":/data/saliency "$box":/data/selections "$box":/data/lm_eval_raw "$dest"/
du -sh "$dest"/* 2>/dev/null
