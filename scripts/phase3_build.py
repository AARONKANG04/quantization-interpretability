"""Build a mixed-precision QDQ checkpoint from a selection file: NVFP4 everywhere except the protected channels.

usage: python scripts/phase3_build.py --selection /data/selections/fg_proj_0010.json --out /data/ckpt/mix_fg_proj_0010
"""
import argparse
import json
from pathlib import Path

import torch

from _common import load_env, run_meta, write_json

load_env()
from qi.models import load_gemma3_text  # noqa: E402
from qi.quant import apply_scheme, summarize_stats  # noqa: E402
from qi.quant.apply import save_qdq_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-pt"); ap.add_argument("--selection", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sel = json.loads(Path(args.selection).read_text())
    protect = {k: {"rows": torch.tensor(v["rows"], dtype=torch.long), "cols": torch.tensor(v["cols"], dtype=torch.long)} for k, v in sel.items()}
    model, tok = load_gemma3_text(args.model)
    for k, v in protect.items():
        v["rows"] = v["rows"].to(model.device); v["cols"] = v["cols"].to(model.device)
    stats = apply_scheme(model, "nvfp4_rtn", protect=protect)
    n_prot = sum(s.get("protected_rows", 0) for s in stats.values()), sum(s.get("protected_cols", 0) for s in stats.values())
    meta = {**run_meta(args), "selection": args.selection, "protected_rows": n_prot[0], "protected_cols": n_prot[1], "summary": summarize_stats(stats)}
    save_qdq_checkpoint(model, tok, args.out, meta)
    write_json(Path("results/phase3/builds") / (Path(args.out).name + ".json"), meta)
    print("built", args.out, "protected rows/cols", n_prot)


if __name__ == "__main__":
    main()
