"""Budgeted channel selections per method, memory accounting, and the whole-layer fallback curve.

usage: python scripts/phase3_select.py --scores /data/saliency/scores_L17.pt --budgets 0.01,0.02,0.05 \
         --ranked results/phase1/ranked_order.json --out results/phase3/selections.json
"""
import argparse
import json
from pathlib import Path

import torch

from _common import load_env, run_meta, write_json

load_env()
from qi.mixed.saliency import layer_fallback_bytes, memory_bytes, select_budget  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True); ap.add_argument("--budgets", default="0.01,0.02,0.05")
    ap.add_argument("--ranked", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--selection-dir", default="/data/selections")
    args = ap.parse_args()
    d = torch.load(args.scores)
    shapes = d["shapes"]
    total = sum(o * i for o, i in shapes.values())
    per_layer = {}
    for name, (o, i) in shapes.items():
        li = int(name.split(".layers.")[1].split(".")[0]); per_layer[li] = per_layer.get(li, 0) + o * i
    layer_weights = total / len(per_layer)
    Path(args.selection_dir).mkdir(parents=True, exist_ok=True)
    out = {**run_meta(args), "total_weights": total, "layer_weights": layer_weights, "selections": []}
    for method, sc in d["scores"].items():
        for b in [float(x) for x in args.budgets.split(",")]:
            budget = int(b * total)
            sel, used = select_budget(sc, shapes, budget)
            f = Path(args.selection_dir) / f"{method}_{int(b * 1000):04d}.json"
            f.write_text(json.dumps(sel))
            mem = memory_bytes(total, used)
            out["selections"].append({"method": method, "budget_frac": b, "protected_weights": used, "n_matrices": len(sel),
                                      "n_rows": sum(len(v["rows"]) for v in sel.values()), "n_cols": sum(len(v["cols"]) for v in sel.values()),
                                      "extra_bytes": mem["extra_bytes"], "bytes_mixed": mem["bytes_mixed"], "file": str(f)})
    if args.ranked:
        rk = json.loads(Path(args.ranked).read_text())
        out["layer_fallback"] = [{"n_layers": n, "extra_bytes": layer_fallback_bytes(layer_weights, n), "layers": rk["orders"]["ranked"][:n]} for n in (1, 2, 3, 5, 8)]
    write_json(args.out, out)
    for s in out["selections"]:
        print(f"{s['method']:8s} {s['budget_frac']:.3f} protected {s['protected_weights'] / total:.4f} rows {s['n_rows']} cols {s['n_cols']} extra {s['extra_bytes'] / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
