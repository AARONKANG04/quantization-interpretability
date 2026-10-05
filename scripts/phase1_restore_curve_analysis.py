"""H1 numbers: accuracy recovered when the N top-ranked layers (ranked by single-layer KL recovery) are restored to bf16,
against random orderings, with paired bootstrap CIs over items. Reads the rc_* evals, writes
results/phase1/restore_curve.json and the ledger.

usage: python scripts/phase1_restore_curve_analysis.py [--ref bf16_rep] [--quant q1_rep] [--raw-root ~/qi_archive/boxA/lm_eval_raw]
"""
import argparse
import re

from _common import (ROOT, bootstrap_module, eval_items, eval_summaries, find_summary, fmt_ci, fmt_pts, load_env,
                     paired_lists, run_meta, set_ledger_row, write_json, write_ledger_block)

load_env()
bs = bootstrap_module()
PAT = re.compile(r"^rc_(ranked|random_s\d+)_top(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="bf16_rep"); ap.add_argument("--quant", default="q1_rep")
    ap.add_argument("--raw-root", default=None); ap.add_argument("--tasks", default="gsm8k_cot,mmlu")
    ap.add_argument("--out", default=str(ROOT / "results/phase1/restore_curve.json"))
    args = ap.parse_args()
    summaries = eval_summaries()
    conds = {}
    for name in summaries:
        m = PAT.match(name)
        if m:
            conds[name] = {"order": m.group(1), "n": int(m.group(2))}
    out = {**run_meta(args), "tasks": {}}
    for task in args.tasks.split(","):
        ref_s, q_s = find_summary(summaries, args.ref, task), find_summary(summaries, args.quant, task)
        if not ref_s or not q_s:
            print(f"{task}: reference or quantised summary missing, skipped"); continue
        ref_items, q_items = eval_items(ref_s, task, args.raw_root), eval_items(q_s, task, args.raw_root)
        if not ref_items or not q_items:
            print(f"{task}: raw samples missing for the reference pair, skipped"); continue
        a, b, _ = paired_lists(ref_items, q_items)
        loss = bs.paired_bootstrap_delta(a, b)  # quant - ref, negative
        T = {"n_items": loss["n"], "acc_ref": loss["acc_a"], "acc_quant": loss["acc_b"], "loss": -loss["delta"],
             "loss_ci": [-loss["hi"], -loss["lo"]], "conditions": {}, "ranked_vs_random": {}}
        items = {}
        for name, c in conds.items():
            s = find_summary(summaries, name, task)
            it = eval_items(s, task, args.raw_root) if s else None
            if not it:
                continue
            items[name] = it
            qa, xb, _ = paired_lists(q_items, it)
            d = bs.paired_bootstrap_delta(qa, xb)  # condition - quant = recovered points
            ra, xb2, _ = paired_lists(ref_items, it)
            dr = bs.paired_bootstrap_delta(ra, xb2)
            L = T["loss"] if T["loss"] > 0 else float("nan")
            T["conditions"][name] = {**c, "acc": d["acc_b"], "recovered_pts": d["delta"], "recovered_ci": [d["lo"], d["hi"]],
                                     "recovery_frac": d["delta"] / L, "recovery_frac_ci": [d["lo"] / L, d["hi"] / L],
                                     "gap_to_ref": dr["delta"], "gap_to_ref_ci": [dr["lo"], dr["hi"]], "n_items": d["n"]}
        for n in sorted({c["n"] for c in conds.values()}):
            rk = f"rc_ranked_top{n}"
            if rk not in items:
                continue
            per_seed = {}
            for name, c in conds.items():
                if c["order"].startswith("random") and c["n"] == n and name in items:
                    ra, rb, _ = paired_lists(items[name], items[rk])
                    d = bs.paired_bootstrap_delta(ra, rb)  # ranked - random
                    per_seed[c["order"]] = {"delta": d["delta"], "ci": [d["lo"], d["hi"]]}
            if per_seed:
                T["ranked_vs_random"][str(n)] = {"per_seed": per_seed,
                                                 "mean_delta": sum(v["delta"] for v in per_seed.values()) / len(per_seed),
                                                 "any_seed_significant": any(v["ci"][0] > 0 for v in per_seed.values())}
        ranked = sorted((c for c in T["conditions"].values() if c["order"] == "ranked"), key=lambda c: c["n"])
        T["n_for_50pct"] = next((c["n"] for c in ranked if c["recovery_frac"] >= 0.5), None)
        T["n_for_80pct"] = next((c["n"] for c in ranked if c["recovery_frac"] >= 0.8), None)
        out["tasks"][task] = T
    # average recovery over GSM8K and MMLU per N (the ledger definition)
    avg = {}
    for n in sorted({c["n"] for c in conds.values()}):
        vals = [out["tasks"][t]["conditions"].get(f"rc_ranked_top{n}", {}).get("recovery_frac") for t in out["tasks"]]
        vals = [v for v in vals if v is not None]
        if vals:
            avg[str(n)] = sum(vals) / len(vals)
    out["avg_recovery_ranked"] = avg
    out["n_for_80pct_avg"] = next((int(n) for n, v in avg.items() if v >= 0.8), None)
    write_json(args.out, out)

    lines = ["", f"## Phase 1: layer restore curve (ref {args.ref}, quant {args.quant}, paired bootstrap, {out['started']})", ""]
    for task, T in out["tasks"].items():
        lines += [f"**{task}**: {args.ref} {100 * T['acc_ref']:.2f}, {args.quant} {100 * T['acc_quant']:.2f}, loss {100 * T['loss']:.2f} "
                  f"{fmt_ci(*T['loss_ci'])} on {T['n_items']} items. 50% recovery at N = {T['n_for_50pct']}, 80% at N = {T['n_for_80pct']}.", "",
                  "| N layers in bf16 | Ranked acc | Recovered pts [95% CI] | Recovery | Random orderings acc | Ranked minus random (mean) |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for n in sorted({c["n"] for c in T["conditions"].values()}):
            rk = T["conditions"].get(f"rc_ranked_top{n}")
            rnd = [c for c in T["conditions"].values() if c["order"].startswith("random") and c["n"] == n]
            rr = T["ranked_vs_random"].get(str(n), {})
            if rk:
                lines.append(f"| {n} | {100 * rk['acc']:.2f} | {fmt_pts(rk['recovered_pts'])} {fmt_ci(*rk['recovered_ci'])} | {100 * rk['recovery_frac']:.0f}% | "
                             + ", ".join(f"{100 * c['acc']:.2f}" for c in sorted(rnd, key=lambda c: c["order"])) + " | "
                             + (f"{fmt_pts(rr['mean_delta'])}" + (" (a seed is significant)" if rr.get("any_seed_significant") else " (n.s.)") if rr else "") + " |")
        lines.append("")
    write_ledger_block("## Phase 1: layer restore curve", lines)
    print("\n".join(lines))
    g = out["tasks"].get("gsm8k_cot")
    if g:
        best = max((c for c in g["conditions"].values() if c["order"] == "ranked"), key=lambda c: c["n"])
        rr = g["ranked_vs_random"].get(str(best["n"]), {})
        rand_recov = ", ".join(f"{100 * c['recovery_frac']:.0f}%" for c in g["conditions"].values()
                               if c["order"].startswith("random") and c["n"] == best["n"])
        reached = "not reached" if out["n_for_80pct_avg"] is None else f"at N = {out['n_for_80pct_avg']}"
        set_ledger_row("[X]% of loss in [N] layers",
                       f"{100 * best['recovery_frac']:.0f}% of the GSM8K loss with the {best['n']} top-ranked layers ({100 * best['n'] / 34:.0f}% of layers); "
                       f"80% recovery {reached}; avg GSM8K+MMLU at N={best['n']}: {100 * avg.get(str(best['n']), float('nan')):.0f}%",
                       fmt_ci(*best["recovery_frac_ci"]).replace("+", ""),
                       f"random orderings recover {rand_recov}; ranked minus random {fmt_pts(rr.get('mean_delta', float('nan')))} pts",
                       "results/phase1/restore_curve.json")


if __name__ == "__main__":
    main()
