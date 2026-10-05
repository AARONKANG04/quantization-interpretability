"""Compare per-feature survival (correlation of activations across models) between conditions at one layer.

usage: python scripts/phase2_compare_controls.py --layer 17 --names q1 c2 [--store-dir /data/shift] --out results/phase2/survival_L17.json
"""
import argparse

import torch

from _common import load_env, write_json

load_env()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--names", nargs="+", default=["q1", "c2"])
    ap.add_argument("--store-dir", default="/data/shift"); ap.add_argument("--min-count", type=int, default=200); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"layer": args.layer, "min_count": args.min_count, "conditions": {}}
    sets = {}
    for name in args.names:
        pf = torch.load(f"{args.store_dir}/{name}_L{args.layer}.pt")["per_feature"]
        corr = pf["corr"].clone(); corr[torch.isnan(corr)] = 1.0
        el = pf["cnt_a"] >= args.min_count
        c = corr[el]
        qs = torch.quantile(c, torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50]))
        low = c < 0.8
        cnt, peak = pf["cnt_a"][el], pf["peak_a"][el]
        d = {"n_eligible": int(el.sum()),
             "n_corr_below": {t: int((c < t).sum()) for t in (0.5, 0.8, 0.95)},
             "frac_corr_below": {t: float((c < t).float().mean()) for t in (0.5, 0.8, 0.95)},
             "corr_quantiles_1_5_10_25_50": [round(float(v), 4) for v in qs],
             "low_survival_median_count": float(cnt[low].median()) if low.any() else None, "all_median_count": float(cnt.median()),
             "low_survival_median_peak": float(peak[low].median()) if low.any() else None, "all_median_peak": float(peak.median()),
             "labels": {k: int(v) for k, v in zip(("suppressed", "amplified", "newly_dead", "newly_alive", "stable", "inactive"),
                                                 [(pf["label"] == v).sum() for v in (0, 1, 2, 3, 4, -1)])}}
        out["conditions"][name] = d
        sets[name] = set(torch.nonzero(el & (corr < 0.5)).flatten().tolist())
        print(name, {k: v for k, v in d.items() if k not in ("labels",)})
    if len(args.names) >= 2:
        a, b = sets[args.names[0]], sets[args.names[1]]
        j = len(a & b) / max(len(a | b), 1)
        out["jaccard_corr_below_0.5"] = j
        print(f"features with corr<0.5 in both {args.names[0]} and {args.names[1]}: {len(a & b)} (Jaccard {j:.3f})")
    write_json(args.out, out)


if __name__ == "__main__":
    main()
