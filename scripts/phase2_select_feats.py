"""Turn shift statistics into feature sets for steering and saliency (plan E2.5 / E3.1).

usage: python scripts/phase2_select_feats.py --shift /data/shift/q1_L17.pt --top-frac 0.01 --seed 0 \
         --out results/phase2/feats_q1_L17.json
Writes: top (by shift energy), top_norm (energy per activation), suppressed subset with gains
(mean active value bf16 / Q1), random control of the same size.
"""
import argparse
import random

import torch

from _common import load_env, run_meta, write_json

load_env()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", required=True); ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--merge", action="store_true", help="keep every set already in --out, only add missing keys")
    args = ap.parse_args()
    d = torch.load(args.shift)
    pf = d["per_feature"]
    energy, cnt_a, cnt_b, label = pf["energy"], pf["cnt_a"], pf["cnt_b"], pf["label"]
    active = (cnt_a + cnt_b) > 0
    n_top = max(1, int(round(args.top_frac * int(active.sum()))))
    order = energy.argsort(descending=True)
    top = [int(i) for i in order[:n_top]]
    norm_energy = energy / (cnt_a + cnt_b).clamp(min=1)
    top_norm = [int(i) for i in norm_energy.argsort(descending=True)[:n_top]]
    gains = (pf["mean_act_a"] / pf["mean_act_b"].clamp(min=1e-6))
    corr = pf["corr"].clone(); corr[torch.isnan(corr)] = 1.0
    eligible = cnt_a >= 200  # enough bf16 activations to estimate a correlation
    surv_score = torch.where(eligible, 1.0 - corr, torch.zeros_like(corr))
    survival = [int(i) for i in surv_score.argsort(descending=True)[:n_top]]  # least-surviving features
    rel = energy / (pf["mean_act_a"].square() * cnt_a).clamp(min=1e-6)  # shift energy relative to the feature's own scale
    rel_energy = [int(i) for i in torch.where(eligible, rel, torch.zeros_like(rel)).argsort(descending=True)[:n_top]]
    suppressed = [i for i in top if label[i] in (0,)]  # frequency ratio < 0.5 in Q1 but still alive
    newly_dead = [i for i in top if label[i] == 2]
    rnd = random.Random(args.seed)
    pool = [int(i) for i in active.nonzero().flatten().tolist() if int(i) not in set(top)]
    control = rnd.sample(pool, n_top)
    # larger sets for the feature-level restore curve (oracle patch): 5% and 10% by survival, by energy, and random
    larger = {}
    for frac in (0.05, 0.10):
        n = int(round(frac * int(active.sum()))); tag = f"{int(frac * 100)}pct"
        larger[f"survival_{tag}"] = [int(i) for i in surv_score.argsort(descending=True)[:n]]
        larger[f"top_{tag}"] = [int(i) for i in order[:n]]
        larger[f"control_{tag}"] = random.Random(args.seed + 1).sample([int(i) for i in active.nonzero().flatten().tolist()], n)
    out = {**run_meta(args), "layer": d["layer"], "sae": d["sae"], "n_active": int(active.sum()), "n_top": n_top,
                          "top": top, "top_norm": top_norm, "survival": survival, "rel_energy": rel_energy, "suppressed": suppressed, "newly_dead": newly_dead, **larger,
                          "gains_top": {str(i): float(gains[i]) for i in set(top) | set(survival) | set(rel_energy)}, "control": control,
                          "survival_corr_threshold": float(corr[survival[-1]]) if survival else None,
                          "n_corr_below_0.5": int(((corr < 0.5) & eligible).sum()), "n_corr_below_0.8": int(((corr < 0.8) & eligible).sum()), "n_eligible": int(eligible.sum()),
                          "energy_share_top": float(energy[top].sum() / energy[active].sum())}
    import json, os
    if args.merge and os.path.exists(args.out):
        old = json.loads(open(args.out).read())
        same = {k: old[k] == out[k] for k in ("top", "survival", "control") if k in old}
        print("merge: existing sets identical to recomputed:", same)
        out = {**out, **{k: v for k, v in old.items() if k not in ("git_sha", "started", "host", "args")}, "merged_from": old.get("started")}
    write_json(args.out, out)
    print({"n_active": int(active.sum()), "n_top": n_top, "suppressed": len(suppressed), "newly_dead": len(newly_dead), "n_corr_below_0.5": int(((corr < 0.5) & eligible).sum()), "survival_thr": float(corr[survival[-1]]),
           "energy_share_top": float(energy[top].sum() / energy[active].sum())})


if __name__ == "__main__":
    main()
