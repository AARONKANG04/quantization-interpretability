"""Regenerate every figure from results JSON. Each figure's title states the finding with its number.

usage: python scripts/figures/make_all.py [--out results/figures]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
style.apply()


def load(rel):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


COND_LABELS = {"q1": "NVFP4", "q1_arc_hs": "NVFP4", "q1_rep": "NVFP4", "q3": "FP8", "q3_arc_hs": "FP8",
               "c1_fp32": "fp32 (noise floor)", "c1_fp32_mmlu": "fp32 (noise floor)"}
TASK_LABELS = {"gsm8k_cot": "GSM8K", "mmlu": "MMLU", "arc_challenge": "ARC-C", "hellaswag": "HellaSwag"}


def fig_damage(out):
    rows = []
    for rel in ("results/phase0/damage_table.json", "results/phase0/damage_table_bf16_rep.json"):
        d = load(rel)
        if not d:
            continue
        for cond, tasks in d["table"].items():
            if cond not in COND_LABELS or (cond == "q1_rep" and rel.endswith("_bf16_rep.json")):
                continue
            for task, e in tasks.items():
                if "paired" in e and not any(c == cond and t == task for c, t, _ in rows):
                    rows.append((cond, task, e["paired"]))
    if not rows:
        return
    order = {"NVFP4": 0, "FP8": 1, "fp32 (noise floor)": 2}
    rows.sort(key=lambda r: (order[COND_LABELS[r[0]]], list(TASK_LABELS).index(r[1])))
    fig, ax = plt.subplots(figsize=(6.4, 0.6 + 0.45 * len(rows)))
    labels = [f"{COND_LABELS[c]} / {TASK_LABELS.get(t, t)}" for c, t, _ in rows]
    deltas = [p["delta"] * 100 for _, _, p in rows]
    err = [[(p["delta"] - p["lo"]) * 100 for _, _, p in rows], [(p["hi"] - p["delta"]) * 100 for _, _, p in rows]]
    colors = [style.ACCENT if COND_LABELS[c] == "NVFP4" else style.MUTED for c, _, _ in rows]
    ax.barh(labels, deltas, xerr=err, color=colors, height=0.55, error_kw={"ecolor": style.INK2, "elinewidth": 1, "capsize": 3})
    ax.axvline(0, color=style.AXIS, lw=0.8); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy change vs bf16 (points), paired bootstrap 95% CI")
    q1 = next((p for c, t, p in rows if c == "q1" and t == "gsm8k_cot"), None)
    style.finish(ax, f"NVFP4 costs {abs(q1['delta']) * 100:.1f} GSM8K points; FP8 and the fp32 reference stay at zero on every suite" if q1 else "Accuracy change vs bf16")
    fig.savefig(out / "f0_damage.png"); plt.close(fig)


def fig_layer_ranking(out):
    d = load("results/phase1/ranked_order.json")
    if not d:
        return
    rk = sorted(d["ranking"], key=lambda r: r["layer"])
    base = d["kl_all_quantized"]
    rec = [100 * r["recovery"] / base for r in rk]
    top = {r["layer"] for r in d["ranking"][:5]}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar([r["layer"] for r in rk], rec, color=[style.ACCENT if r["layer"] in top else style.MUTED for r in rk], width=0.8)
    ax.set_xlabel("layer restored to bf16 (all others NVFP4)"); ax.set_ylabel("% of total KL removed")
    ax.set_xticks(range(0, 34, 2))
    for r in d["ranking"][:5]:
        ax.annotate(str(r["layer"]), (r["layer"], 100 * r["recovery"] / base), ha="center", va="bottom", fontsize=8, color=style.INK)
    t1 = 100 * d["ranking"][0]["recovery"] / base
    style.finish(ax, f"Every layer contributes: the most damaging single layer removes only {t1:.1f}% of the NVFP4 KL",
                 "Each bar: one layer back in bf16, the other 33 in NVFP4; KL(bf16 || model) on the 500k-token held-out set.")
    fig.savefig(out / "f1_layer_ranking.png"); plt.close(fig)


def fig_tokens(out):
    d = load("results/phase1/token_buckets_q1.json")
    if not d:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, key, title in ((axes[0], "by_class", "token class"), (axes[1], "by_entropy_quintile", "bf16 entropy quintile"), (axes[2], "by_position", "position in context")):
        b = d[key]["buckets"]
        names = [n for n, v in b.items() if v["n"] >= 1000]
        order = {"by_entropy_quintile": ["q1_lowest", "q2", "q3", "q4", "q5_highest"], "by_position": ["0-256", "256-1024", "1024-4096", "4096+"]}.get(key)
        if order:
            names = [n for n in order if n in names]
        else:
            names = sorted(names, key=lambda n: -b[n]["flip_enrichment"])
        vals = [b[n]["flip_enrichment"] for n in names]
        colors = [style.ACCENT if v == max(vals) else style.MUTED for v in vals]
        ax.bar(names, vals, color=colors, width=0.7)
        ax.axhline(1, color=style.AXIS, lw=0.8, ls="--")
        ax.set_ylabel("flip rate / overall flip rate"); ax.set_xlabel(title)
        ax.tick_params(axis="x", rotation=30)
    e5 = d["by_entropy_quintile"]["buckets"].get("q5_highest", {}).get("flip_enrichment", 0)
    fig.suptitle(f"Top-1 flips concentrate where bf16 is uncertain: {e5:.1f}x the average rate in the highest-entropy fifth", x=0.01, ha="left", fontsize=11, fontweight="semibold", color=style.INK)
    fig.tight_layout()
    fig.savefig(out / "f2_token_buckets.png"); plt.close(fig)


def fig_restore_curve(out):
    d = load("results/phase1/restore_curve.json")
    if not d or "gsm8k_cot" not in d["tasks"]:
        return
    T = d["tasks"]["gsm8k_cot"]
    ranked = sorted((c for c in T["conditions"].values() if c["order"] == "ranked"), key=lambda c: c["n"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    seeds = sorted({c["order"] for c in T["conditions"].values() if c["order"].startswith("random")})
    for i, sd in enumerate(seeds):
        pts = sorted((c for c in T["conditions"].values() if c["order"] == sd), key=lambda c: c["n"])
        ax.plot([c["n"] for c in pts], [100 * c["recovery_frac"] for c in pts], color=style.MUTED, marker="o", lw=1.5,
                label="random orderings" if i == 0 else None)
    xs = [c["n"] for c in ranked]; ys = [100 * c["recovery_frac"] for c in ranked]
    err = [[100 * (c["recovery_frac"] - c["recovery_frac_ci"][0]) for c in ranked], [100 * (c["recovery_frac_ci"][1] - c["recovery_frac"]) for c in ranked]]
    ax.errorbar(xs, ys, yerr=err, color=style.ACCENT, marker="o", lw=2, capsize=3, label="ranked by single-layer KL")
    ax.axhline(100, color=style.AXIS, lw=0.8, ls="--"); ax.axhline(0, color=style.AXIS, lw=0.8)
    ax.set_xlabel("layers restored to bf16 (of 34)"); ax.set_ylabel("% of GSM8K loss recovered"); ax.set_xticks(xs)
    ax.set_ylim(-20, 110); ax.legend(loc="upper left")
    last = ranked[-1]
    style.finish(ax, f"Restoring the {last['n']} most damaging layers recovers {100 * last['recovery_frac']:.0f}% of the loss; a random ordering does as well",
                 f"Paired bootstrap 95% CIs on {T['n_items']} items; loss = bf16 minus NVFP4 = {100 * T['loss']:.1f} points.")
    fig.savefig(out / "f3_restore_curve.png"); plt.close(fig)


def fig_budget_sweep(out):
    d = load("results/phase3/budget_sweep.json")
    if not d or "gsm8k_cot" not in d["tasks"]:
        return
    T = d["tasks"]["gsm8k_cot"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    names = {"fg_proj": "feature-guided", "mag": "magnitude", "klg": "KL-gradient", "rand": "random channels"}
    for i, m in enumerate(["fg_proj", "mag", "klg", "rand"]):
        pts = sorted((c for c in T["conditions"].values() if c["method"] == m), key=lambda c: c["budget"])
        if not pts:
            continue
        err = [[100 * (c["recovery_frac"] - c["recovery_frac_ci"][0]) for c in pts], [100 * (c["recovery_frac_ci"][1] - c["recovery_frac"]) for c in pts]]
        ax.errorbar([c["extra_mb"] for c in pts], [100 * c["recovery_frac"] for c in pts], yerr=err, color=style.SERIES[i], marker="o", lw=2,
                    capsize=2, elinewidth=0.8, label=names[m], alpha=0.95 if m in ("fg_proj", "mag") else 0.75, errorevery=1)
        for line in ax.containers[-1].lines[2]:  # lighten the CI whiskers so the means stay readable
            line.set_alpha(0.35)
    lf = sorted(T["layer_fallback"].values(), key=lambda c: c["n_layers"])
    if lf:
        ax.plot([c["extra_mb"] for c in lf], [100 * c["recovery_frac"] for c in lf], color=style.INK2, marker="s", lw=1.5, ls="--", label="whole-layer fallback")
        for c in lf:
            ax.annotate(f"{c['n_layers']}L", (c["extra_mb"], 100 * c["recovery_frac"]), textcoords="offset points", xytext=(4, -10), fontsize=7, color=style.INK2)
    ax.set_xscale("log"); ax.set_xlabel("extra memory over NVFP4 (MB, log scale)"); ax.set_ylabel("% of GSM8K loss recovered")
    ax.axhline(0, color=style.AXIS, lw=0.8); ax.legend(loc="upper left", ncol=2)
    mag2 = next((c for c in T["conditions"].values() if c["method"] == "mag" and abs(c["budget"] - 0.02) < 1e-6), None)
    l8 = next((c for c in lf if c["n_layers"] == 8), None)
    title = (f"Magnitude channels at {mag2['extra_mb']:.0f} MB recover {100 * mag2['recovery_frac']:.0f}%, as much as 8 whole layers at {l8['extra_mb']:.0f} MB; feature guidance ties"
             if mag2 and l8 else "Recovery vs extra memory")
    style.finish(ax, title, "Paired bootstrap 95% CIs vs NVFP4 on 1319 GSM8K items; budgets 1%, 2%, 5% of weights in bf16.")
    fig.savefig(out / "f4_budget_sweep.png"); plt.close(fig)


def fig_steering(out):
    d = load("results/phase2/steering.json")
    if not d:
        return
    order = [("gain", "gain correction, 1% least-surviving"), ("gain_random", "gain correction, 1% random (control)"), ("gain_top", "gain correction, 1% highest-energy"),
             ("mean_shift", "mean-shift bias (deployable)"), ("oracle", "oracle patch, 1% least-surviving"), ("oracle_random", "oracle patch, 1% random (control)"),
             ("oracle_survival_10pct", "oracle patch, 10% least-surviving"), ("oracle_control_10pct", "oracle patch, 10% random (control)"),
             ("oracle_top_10pct", "oracle patch, 10% highest-energy"), ("oracle_all", "oracle patch, all 65k features"), ("full_residual", "replace whole residual (upper bound)")]
    rows = [(lab, d["arms"][k]) for k, lab in order if k in d["arms"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.6 + 0.42 * len(rows)))
    y = list(range(len(rows)))[::-1]
    vals = [100 * a["delta_vs_none"] for _, a in rows]
    err = [[100 * (a["delta_vs_none"] - a["ci"][0]) for _, a in rows], [100 * (a["ci"][1] - a["delta_vs_none"]) for _, a in rows]]
    colors = [style.ACCENT if a["significant"] else style.MUTED for _, a in rows]
    ax.barh(y, vals, xerr=err, color=colors, height=0.6, error_kw={"ecolor": style.INK2, "elinewidth": 1, "capsize": 3})
    ax.set_yticks(y); ax.set_yticklabels([lab for lab, _ in rows]); ax.axvline(0, color=style.AXIS, lw=0.8)
    gap = (d.get("ref_acc_same_items") or 0) - d["baseline_acc"]
    if gap > 0:
        ax.axvline(100 * gap, color=style.INK2, lw=0.8, ls="--"); ax.text(100 * gap, len(rows) - 0.4, " bf16 gap", fontsize=8, color=style.INK2, va="bottom")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False); ax.set_xlabel("GSM8K accuracy change vs no steering (points), 500 items, paired 95% CI")
    top = d["arms"].get("oracle_top_10pct"); allf = d["arms"].get("oracle_all")
    title = (f"No 1% feature set carries the loss; the 10% highest-energy features recover {100 * top.get('recovery_frac_of_subset_gap', 0):.0f}%, as much as all features"
             if top and allf else "Interventions at layer 17")
    style.finish(ax, title)
    fig.savefig(out / "f5_steering.png"); plt.close(fig)


def fig_survival(out):
    d17, d29 = load("results/phase2/survival_L17.json"), load("results/phase2/survival_L29.json")
    if not d17:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    groups, vals, cols = [], [], []
    for layer, d in ((17, d17), (29, d29)):
        if not d:
            continue
        for cond, name, col in (("q1", "NVFP4", style.ACCENT), ("c2", "matched noise", style.SERIES[1])):
            c = d["conditions"].get(cond)
            if c:
                groups.append(f"layer {layer}\n{name}"); vals.append(100 * c["frac_corr_below"]["0.8"]); cols.append(col)
    ax.bar(groups, vals, color=cols, width=0.6)
    for x, v in enumerate(vals):
        ax.annotate(f"{v:.0f}%", (x, v), ha="center", va="bottom", fontsize=8, color=style.INK)
    ax.set_ylabel("% of features with survival corr. < 0.8")
    q = d17["conditions"]["q1"]["frac_corr_below"]["0.8"]; n = d17["conditions"]["c2"]["frac_corr_below"]["0.8"]
    style.finish(ax, f"NVFP4 damages fewer SAE features than Gaussian noise of the same size: {100 * q:.0f}% vs {100 * n:.0f}% at layer 17",
                 "Gemma Scope 2 65k features with at least 200 bf16 activations. Survival: correlation of a feature's activations across the two models.")
    fig.savefig(out / "f6_survival.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results/figures")); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for f in (fig_damage, fig_layer_ranking, fig_tokens, fig_restore_curve, fig_budget_sweep, fig_steering, fig_survival):
        try:
            f(out)
        except Exception as e:  # noqa: BLE001
            print(f"{f.__name__}: {type(e).__name__}: {e}")
    print("figures:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
