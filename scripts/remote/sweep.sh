#!/usr/bin/env bash
# Phase 1 E1.3 sweep 2 on the 500k-token subset: fully quantized baseline, bf16 sanity, restore-one-layer x 34.
cd ~/quantization-interpretability || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/data/hf TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
set -a; . ./.env; set +a
T=/data/kl/kl_500k.pt; R=/data/kl/ref_kl_500k_top256.pt; O=results/phase1/layer_sweep
[ -e $R ] || uv run python scripts/phase1_layer_sweep.py --build-ref --tokens $T --ref $R --batch 2
[ -e /data/kl/ref_kl_2m_top256.pt ] || uv run python scripts/phase1_layer_sweep.py --build-ref --tokens /data/kl/kl_2m.pt --ref /data/kl/ref_kl_2m_top256.pt --batch 2
uv run python scripts/phase1_layer_sweep.py --mode none        --tokens $T --ref $R --batch 2 --out $O
uv run python scripts/phase1_layer_sweep.py --mode all         --tokens $T --ref $R --batch 2 --out $O
uv run python scripts/phase1_layer_sweep.py --mode restore_one --layers ${1:-0-33} --tokens $T --ref $R --batch 2 --out $O
