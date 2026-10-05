# One target per phase; every target is a thin wrapper over scripts/ so runs are reproducible from the shell.
PY ?= uv run python
GPU ?= 0

.PHONY: env test lint smoke phase0 phase1 figures

env:
	uv sync --extra dev
	@echo "On the box: uv sync --extra dev --extra box"

test:
	uv run pytest -q

lint:
	uv run ruff check qi scripts tests

smoke:
	$(PY) scripts/smoke.py

# Phase 0: build the QDQ checkpoints, evaluate every condition, write the damage table.
phase0:
	$(PY) scripts/phase0_pilot.py --scheme nvfp4_rtn --out /data/ckpt/q1 --stats-out results/phase0/pilot/q1.json
	$(PY) scripts/phase0_pilot.py --scheme fp8_e4m3 --out /data/ckpt/q3 --stats-out results/phase0/pilot/q3.json
	$(PY) scripts/phase0_pilot.py --scheme noise_matched --out /data/ckpt/c2 --stats-out results/phase0/pilot/c2.json
	@echo "Now run scripts/phase0_eval.py for bf16, fp32, q1, q3 (one GPU each; see scripts/remote/run.sh)"

# Phase 1: restore-one-layer sweep on the 500k-token subset (shard --layers across GPUs).
phase1:
	$(PY) scripts/phase1_layer_sweep.py --mode restore_one --layers 0-33 --out results/phase1/layer_sweep

figures:
	$(PY) scripts/figures/make_all.py
