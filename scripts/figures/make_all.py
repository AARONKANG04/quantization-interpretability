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


def fig_damage(out):
    d = load("results/phase0/damage_table.json")
    if not d:
        return
    rows = []
    for cond, tasks in d["table"].items():
        for task, e in tasks.items():
            if "paired" in e:
                rows.append((cond, task, e["paired"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.6 + 0.5 * len(rows)))
    labels = [f"{c} / {t}" for c, t, _ in rows]
    deltas = [p["delta"] * 100 for _, _, p in rows]
    err = [[(p["delta"] - p["lo"]) * 100 for _, _, p in rows], [(p["hi"] - p["delta"]) * 100 for _, _, p in rows]]
    colors = [style.ACCENT if c == "q1" else style.MUTED for c, _, _ in rows]
    ax.barh(labels, deltas, xerr=err, color=colors, height=0.55, error_kw={"ecolor": style.INK2, "elinewidth": 1, "capsize": 3})
    ax.axvline(0, color=style.AXIS, lw=0.8); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy change vs bf16 (points), paired bootstrap 95% CI")
    q1 = next((p for c, t, p in rows if c == "q1" and t == "gsm8k_cot"), None)
    style.finish(ax, f"NVFP4 costs {abs(q1['delta']) * 100:.1f} GSM8K points; FP8 and the fp32 reference stay near zero" if q1 else "Accuracy change vs bf16")
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
    t5 = sum(100 * r["recovery"] / base for r in d["ranking"][:5])
    style.finish(ax, f"Restoring the top 5 layers alone removes {t5:.0f}% of the NVFP4 KL (single-layer effects, 500k tokens)",
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results/figures")); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for f in (fig_damage, fig_layer_ranking, fig_tokens):
        try:
            f(out)
        except Exception as e:  # noqa: BLE001
            print(f"{f.__name__}: {type(e).__name__}: {e}")
    print("figures:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
