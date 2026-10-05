"""Rank layers by single-layer marginal KL recovery (from the restore-one sweep) and build the checkpoints for
the accuracy curve: top-N ranked layers restored to bf16 for N in --ns, plus a random ordering per seed.

usage: python scripts/phase1_ranked_curve.py --sweep results/phase1/layer_sweep --ckpt-dir /data/ckpt \
         [--ns 1,2,3,5,8] [--random-seeds 0] [--scheme nvfp4_rtn]
Then evaluate each checkpoint with scripts/phase0_eval.py (one GPU each on boxA).
"""
import argparse
import glob
import json
import random
from pathlib import Path

from _common import load_env, run_meta, write_json

load_env()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/phase1/layer_sweep")
    ap.add_argument("--ckpt-dir", default="/data/ckpt")
    ap.add_argument("--ns", default="1,2,3,5,8")
    ap.add_argument("--random-seeds", default="0")
    ap.add_argument("--scheme", default="nvfp4_rtn")
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    ap.add_argument("--out", default="results/phase1/ranked_order.json")
    ap.add_argument("--dry", action="store_true", help="only write the ordering, build no checkpoints")
    ap.add_argument("--skip-ranked", action="store_true", help="build only the random orderings")
    args = ap.parse_args()

    base = json.loads(Path(args.sweep, "all_x.json").read_text())["mean_kl"]
    rows = []
    for f in glob.glob(str(Path(args.sweep) / "restore_one_*.json")):
        d = json.loads(Path(f).read_text())
        rows.append({"layer": d["layer"], "kl_restored": d["mean_kl"], "recovery": base - d["mean_kl"]})
    rows.sort(key=lambda r: -r["recovery"])
    ranked = [r["layer"] for r in rows]
    n_layers = len(ranked)
    ns = [int(x) for x in args.ns.split(",")]
    orders = {} if args.skip_ranked else {"ranked": ranked}
    for seed in [int(s) for s in args.random_seeds.split(",")]:
        rnd = list(range(n_layers)); random.Random(seed).shuffle(rnd)
        orders[f"random_s{seed}"] = rnd
    plan = []
    for name, order in orders.items():
        for n in ns:
            plan.append({"order": name, "n": n, "restored": order[:n], "ckpt": str(Path(args.ckpt_dir) / f"{name}_top{n}")})
    write_json(args.out, {**run_meta(args), "kl_all_quantized": base, "ranking": rows, "orders": {"ranked": ranked, **orders}, "plan": plan})
    if args.dry:
        return
    import torch
    from qi.models import load_gemma3_text
    from qi.quant import apply_scheme, iter_quantizable
    from qi.quant.apply import save_qdq_checkpoint, summarize_stats
    model, tok = load_gemma3_text(args.model, device_map="cuda" if torch.cuda.is_available() else "cpu")
    originals = {n: m.weight.data.to("cpu", copy=True) for n, m in iter_quantizable(model)}
    for p in plan:
        if Path(p["ckpt"], "config.json").exists():
            print("exists", p["ckpt"]); continue
        for n, m in iter_quantizable(model):
            m.weight.data.copy_(originals[n])
        stats = apply_scheme(model, args.scheme, exclude_layers=p["restored"])
        save_qdq_checkpoint(model, tok, p["ckpt"], {"restored": p["restored"], "order": p["order"], "summary": summarize_stats(stats)})
        print("built", p["ckpt"], "restored", p["restored"], flush=True)


if __name__ == "__main__":
    main()
