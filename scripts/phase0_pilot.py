"""Build one QDQ checkpoint (bf16 safetensors) and write its per-matrix statistics.

usage: python scripts/phase0_pilot.py --scheme nvfp4_rtn --out /data/ckpt/q1 --stats-out results/phase0/pilot/q1.json
       [--layers 0,1,2] [--exclude-layers 5] [--matrices down_proj] [--seed 0] [--model google/gemma-3-4b-pt]
"""
import argparse

import torch

from _common import load_env, run_meta, write_json

load_env()
from qi.models import load_gemma3_text  # noqa: E402
from qi.quant import SCHEMES, apply_scheme, summarize_stats  # noqa: E402
from qi.quant.apply import save_qdq_checkpoint  # noqa: E402


def _ints(s):
    if not s:
        return None
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=SCHEMES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", required=True)
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    ap.add_argument("--layers", default=None, help="comma list or ranges, e.g. 0-4,9")
    ap.add_argument("--exclude-layers", default=None)
    ap.add_argument("--matrices", default=None, help="comma list of suffixes, e.g. down_proj,o_proj")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    model, tok = load_gemma3_text(args.model, device_map="cuda" if torch.cuda.is_available() else "cpu")
    stats = apply_scheme(model, args.scheme, layers=_ints(args.layers), exclude_layers=_ints(args.exclude_layers),
                         matrices=args.matrices.split(",") if args.matrices else None, seed=args.seed)
    summary = summarize_stats(stats)
    meta = {**run_meta(args), "summary": summary}
    if not args.no_save:
        save_qdq_checkpoint(model, tok, args.out, meta)
    write_json(args.stats_out, {**meta, "per_matrix": stats})
    print("summary:", {k: v for k, v in summary.items() if k != "layer_mse_rel"})


if __name__ == "__main__":
    main()
