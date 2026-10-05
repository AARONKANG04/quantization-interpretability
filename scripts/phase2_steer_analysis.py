"""H3 intervention numbers: every steering arm against the no-steering baseline on the same GSM8K items (paired
bootstrap), the feature-level restore curve (oracle patches of growing feature sets up to the whole SAE basis),
and the bf16 ceiling on the same items. Writes results/phase2/steering.json and the ledger.

usage: python scripts/phase2_steer_analysis.py [--layer 17] [--ref bf16_rep] [--quant q1_rep] [--raw-root ~/qi_archive/boxA/lm_eval_raw]
"""
import argparse
import glob
import json
from pathlib import Path

from _common import (ROOT, bootstrap_module, eval_items, eval_summaries, find_summary, fmt_ci, fmt_pts, load_env,
                     paired_lists, run_meta, set_ledger_row, write_json, write_ledger_block)

load_env()
bs = bootstrap_module()


def strict_items(d: dict) -> dict:
    """doc_id -> strict correctness. Older runs stored one row per (doc, filter) without the filter name; lm-eval
    emits all strict rows first, which is verified against the reported strict accuracy before trusting the split."""
    items, n = d["items"], d["metrics"]["sample_len"]
    if items and items[0].get("flexible") is not None and len(items) == n:
        return {it["doc_id"]: bool(it["strict"]) for it in items}
    if len(items) == 2 * n:
        first = items[:n]
        if abs(sum(float(it["strict"]) for it in first) / n - d["metrics"]["exact_match,strict-match"]) < 1e-6 \
                and {it["doc_id"] for it in first} == {it["doc_id"] for it in items[n:]}:
            return {it["doc_id"]: bool(it["strict"]) for it in first}
    raise ValueError(f"cannot recover strict per-item results for {d.get('mode')}/{d.get('set')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17); ap.add_argument("--steer-dir", default=str(ROOT / "results/phase2/steer"))
    ap.add_argument("--ref", default="bf16_rep"); ap.add_argument("--quant", default="q1_rep"); ap.add_argument("--raw-root", default=None)
    ap.add_argument("--out", default=str(ROOT / "results/phase2/steering.json"))
    args = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(f"{args.steer_dir}/q1_L{args.layer}_*.json")):
        d = json.loads(Path(f).read_text())
        key = Path(f).stem[len(f"q1_L{args.layer}_"):]
        runs[key] = {"mode": d["mode"], "set": d["set"], "n_feats": d["n_feats"], "acc": d["metrics"]["exact_match,strict-match"],
                     "n": d["metrics"]["sample_len"], "items": strict_items(d)}
    base = runs.get("none")
    if not base:
        raise SystemExit("no baseline run (mode none)")
    out = {**run_meta(args), "layer": args.layer, "baseline_acc": base["acc"], "n_items": base["n"], "arms": {}}
    # bf16 and quantised accuracy on the same items from the vLLM evals (different engine, so a reference, not a pair)
    summaries = eval_summaries()
    docs = set(base["items"])
    for label, name in (("ref", args.ref), ("quant", args.quant)):
        s = find_summary(summaries, name, "gsm8k_cot")
        it = eval_items(s, "gsm8k_cot", args.raw_root) if s else None
        if it:
            sub = [v for (task, doc), v in it.items() if doc in docs]
            out[f"{label}_acc_same_items"] = sum(sub) / len(sub) if sub else None
            out[f"{label}_acc_full"] = sum(it.values()) / len(it)
    for key, r in runs.items():
        if key == "none" or r["n"] != base["n"]:
            continue
        a, b, _ = paired_lists(base["items"], r["items"])
        d = bs.paired_bootstrap_delta(a, b)
        out["arms"][key] = {"mode": r["mode"], "set": r["set"], "n_feats": r["n_feats"], "acc": r["acc"],
                            "delta_vs_none": d["delta"], "ci": [d["lo"], d["hi"]], "significant": d["lo"] > 0 or d["hi"] < 0, "n": d["n"]}
    # full-set sanity: unhooked HF run vs the vLLM quantised eval on the same docs
    nf = runs.get("none_full")
    if nf:
        s = find_summary(summaries, args.quant, "gsm8k_cot")
        it = eval_items(s, "gsm8k_cot", args.raw_root) if s else None
        if it:
            q = {doc: v for (task, doc), v in it.items()}
            a, b, _ = paired_lists(q, nf["items"])
            d = bs.paired_bootstrap_delta(a, b)
            out["engine_check_hf_minus_vllm_full_set"] = {"delta": d["delta"], "ci": [d["lo"], d["hi"]], "n": d["n"], "acc_hf": d["acc_b"], "acc_vllm": d["acc_a"]}
    ceiling = out.get("ref_acc_same_items")
    if ceiling is not None:
        for a in out["arms"].values():
            a["recovery_frac_of_subset_gap"] = a["delta_vs_none"] / (ceiling - base["acc"]) if ceiling > base["acc"] else None
    write_json(args.out, out)

    ORDER = ["gain", "gain_random", "gain_top", "mean_shift", "oracle", "oracle_random", "oracle_top", "oracle_survival_5pct", "oracle_control_5pct",
             "oracle_top_5pct", "oracle_survival_10pct", "oracle_control_10pct", "oracle_top_10pct", "oracle_all", "full_residual", "bf16_gain"]
    lines = ["", f"## Phase 2: steering at layer {args.layer} (GSM8K subset, {base['n']} items, paired bootstrap vs no steering, {out['started']})", "",
             f"No steering (NVFP4, HF engine): {100 * base['acc']:.1f}. Same items, vLLM engine: bf16 {100 * (out.get('ref_acc_same_items') or float('nan')):.1f}, "
             f"NVFP4 {100 * (out.get('quant_acc_same_items') or float('nan')):.1f}.", "",
             "| Arm | Feature set | Features | Acc | Delta vs none [95% CI] | Share of bf16 gap |", "| --- | --- | --- | --- | --- | --- |"]
    for key in ORDER + sorted(set(out["arms"]) - set(ORDER)):
        a = out["arms"].get(key)
        if a:
            share = a.get("recovery_frac_of_subset_gap")
            lines.append(f"| {a['mode']} | {a['set']} | {a['n_feats']} | {100 * a['acc']:.1f} | {fmt_pts(a['delta_vs_none'])} {fmt_ci(*a['ci'])}"
                         f"{' *' if a['significant'] else ''} | {'' if share is None else f'{100 * share:.0f}%'} |")
    ec = out.get("engine_check_hf_minus_vllm_full_set")
    if ec:
        lines += ["", f"Engine check on the full set: HF backend {100 * ec['acc_hf']:.2f} vs vLLM {100 * ec['acc_vllm']:.2f}, delta {fmt_pts(ec['delta'])} {fmt_ci(*ec['ci'])}."]
    lines.append("")
    write_ledger_block("## Phase 2: steering", lines)
    print("\n".join(lines))
    gain, ms = out["arms"].get("gain"), out["arms"].get("mean_shift")
    full_gap = (out.get("ref_acc_full") or 0) - (out.get("quant_acc_full") or 0)
    if gain:
        set_ledger_row("steering recovered [X] of [Y] points",
                       f"{fmt_pts(gain['delta_vs_none'])} of {100 * full_gap:.1f} points with gain correction on the {gain['n_feats']} least-surviving features"
                       f"{' (n.s.)' if not gain['significant'] else ''}" + (f"; mean-shift bias {fmt_pts(ms['delta_vs_none'])} {fmt_ci(*ms['ci'])}" if ms else ""),
                       fmt_ci(*gain["ci"]),
                       "random-feature gain " + (fmt_pts(out["arms"]["gain_random"]["delta_vs_none"]) if "gain_random" in out["arms"] else "n/a")
                       + "; oracle 1% " + (fmt_pts(out["arms"]["oracle"]["delta_vs_none"]) if "oracle" in out["arms"] else "n/a")
                       + "; full residual " + (fmt_pts(out["arms"]["full_residual"]["delta_vs_none"]) if "full_residual" in out["arms"] else "n/a"),
                       "results/phase2/steering.json")


if __name__ == "__main__":
    main()
