"""E1.4 analysis: per-class / per-position KL and flip statistics with document-cluster bootstrap CIs and
Benjamini-Hochberg-corrected enrichment tests, from the token-dump parquet files.

usage: python scripts/phase1_token_analysis.py --dumps results/phase1/token_dump/q1_kl_2m.parquet \
         results/phase1/token_dump/q1_gsm8k_test.parquet --out results/phase1/token_buckets_q1.json
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_env, run_meta, write_json

load_env()
from qi.metrics.bootstrap import cluster_bootstrap_mean  # noqa: E402


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(adj, 0, 1)
    return out


MIN_N = 1000


def bucket_table(df, col, n_boot=400):
    rows = {}
    base_flip = 1 - df.top1_agree.mean()
    base_median = df.kl.median()
    pvals, names = [], []
    for name, g in df.groupby(col, observed=True):
        flip = 1 - g.top1_agree.mean()
        kl_ci = cluster_bootstrap_mean(g.kl.values, g.doc_id.values, n_boot=n_boot)
        # one-sided binomial-style z test for flip-rate enrichment against the pooled rate
        n = len(g); se = np.sqrt(base_flip * (1 - base_flip) / max(n, 1)) + 1e-12
        z = (flip - base_flip) / se
        from scipy.stats import norm
        p = float(norm.sf(z))
        rows[str(name)] = {"n": int(n), "median_kl": float(g.kl.median()), "median_kl_ratio": float(g.kl.median() / max(base_median, 1e-12)),
                           "mean_kl": kl_ci["mean"], "mean_kl_lo": kl_ci["lo"], "mean_kl_hi": kl_ci["hi"],
                           "p90_kl": float(g.kl.quantile(0.9)), "flip_rate": float(flip), "flip_enrichment": float(flip / max(base_flip, 1e-12)), "p_raw": p}
        pvals.append(p); names.append(str(name))
    adj = bh(pvals) if pvals else []
    for nm, q in zip(names, adj):
        rows[nm]["p_bh"] = float(q); rows[nm]["significant_5pct"] = bool(q < 0.05 and rows[nm]["n"] >= MIN_N)
    return {"base_flip_rate": float(base_flip), "base_median_kl": float(base_median), "buckets": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--tokenizer", default="google/gemma-3-4b-pt")
    ap.add_argument("--min-count", type=int, default=200)
    args = ap.parse_args()
    frames = []
    for f in args.dumps:
        d = pd.read_parquet(f); d["dump"] = Path(f).stem; frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # make doc ids unique across dumps
    df["doc_id"] = df["dump"].astype(str) + ":" + df["doc_id"].astype(str)
    out = {**run_meta(args), "n_tokens": int(len(df)), "overall": {"mean_kl": float(df.kl.mean()), "median_kl": float(df.kl.median()),
           "p90_kl": float(df.kl.quantile(0.9)), "flip_rate": float(1 - df.top1_agree.mean())}}
    out["by_class"] = bucket_table(df, "token_class", args.n_boot)
    out["by_position"] = bucket_table(df, "position_bucket", args.n_boot)
    out["by_dump"] = bucket_table(df, "dump", args.n_boot)
    # class x position interaction (position effect holding class fixed), words and numerals only
    sub = df[df.token_class.isin(["word", "numeral"])]
    out["class_by_position"] = {c: bucket_table(g, "position_bucket", args.n_boot)["buckets"] for c, g in sub.groupby("token_class", observed=True)}
    ent = pd.qcut(df.entropy_a, 5, labels=["q1_lowest", "q2", "q3", "q4", "q5_highest"], duplicates="drop")
    df["entropy_q"] = ent
    out["by_entropy_quintile"] = bucket_table(df, "entropy_q", args.n_boot)
    # class effects within entropy quintiles: is a class still enriched once uncertainty is held fixed?
    out["class_within_entropy"] = {str(q): bucket_table(g, "token_class", max(args.n_boot // 4, 50))["buckets"] for q, g in df.groupby("entropy_q", observed=True)}
    # the individual tokens that flip most, decoded (at least --min-count occurrences)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        g = df.groupby("target").agg(n=("kl", "size"), flip=("top1_agree", lambda v: float(1 - v.mean())), median_kl=("kl", "median"))
        g = g[g.n >= args.min_count].sort_values("flip", ascending=False).head(40)
        out["top_flip_tokens"] = [{"token": tok.convert_ids_to_tokens(int(i)), "n": int(r.n), "flip_rate": float(r.flip), "median_kl": float(r.median_kl)} for i, r in g.iterrows()]
    except Exception as e:  # noqa: BLE001
        out["top_flip_tokens_error"] = str(e)[:200]
    write_json(args.out, out)
    cls = out["by_class"]["buckets"]
    top = sorted([kv for kv in cls.items() if kv[1]["n"] >= MIN_N], key=lambda kv: -kv[1]["flip_enrichment"])[:3]
    print("flip enrichment by class:", [(k, round(v["flip_enrichment"], 2), v["significant_5pct"]) for k, v in top])
    print("median KL ratio by class:", {k: round(v["median_kl_ratio"], 2) for k, v in cls.items()})
    print("top flip tokens:", [(t["token"], round(t["flip_rate"], 2)) for t in out.get("top_flip_tokens", [])[:12]])


if __name__ == "__main__":
    main()
